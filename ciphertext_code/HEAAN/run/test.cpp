#include "../src/HEAAN.h"
#include "../src/TestScheme.h"

#include <NTL/BasicThreadPool.h>
#include <NTL/ZZ.h>

#include "../src/Ciphertext.h"
#include "../src/EvaluatorUtils.h"
#include "../src/Ring.h"
#include "../src/Scheme.h"
#include "../src/SchemeAlgo.h"
#include "../src/SecretKey.h"
#include "../src/StringUtils.h"
#include "../src/TimeUtils.h"
#include "../src/SerializationUtils.h"
#include <unistd.h>
#include <sys/resource.h>
#include <bits/stdc++.h>
#include <omp.h>
using namespace std;





/**
 * Returns the current resident set size (physical memory use) measured
 * in bytes, or zero if the value cannot be determined on this OS.
 */
size_t getCurrentRSS()
{

    /* Linux ---------------------------------------------------- */
    long rss = 0L;
    FILE* fp = NULL;
    if ( (fp = fopen( "/proc/self/statm", "r" )) == NULL )
        return (size_t)0L;      /* Can't open? */
    if ( fscanf( fp, "%*s%ld", &rss ) != 1 )
    {
        fclose( fp );
        return (size_t)0L;      /* Can't read? */
    }
    fclose( fp );
    return (size_t)rss * (size_t)sysconf( _SC_PAGESIZE);

    /* AIX, BSD, Solaris, and Unknown OS ------------------------ */
    return (size_t)0L;          /* Unsupported. */

}






// 网络结构的参数
int in_chanels[]={1,32,32,64,128,128};
int out_chanels[]={32,32,64,128,128,192};
int kernels[] ={13,11,9,7,5,3};
int fc_in[]={192*9,256};
int fc_out[]={256,10};
double dp_rate = 0.7;
//pooling setting
int stride = 3;
int kernel = 3;

//加密方案的参数
long logq = 1200; ///< Ciphertext Modulus
long logp = 30; ///< Real message will be quantized by multiplying 2^40
long logn = 10; ///< log2(The number of slots)
long logpc = 30;
long n = 1 << logn;



//所有函数的返回值都是void，最好别返回带有ciphertext的相关值，可能会错。
void reshape(vector <vector<Ciphertext > >&data,vector<Ciphertext>& res);

//测试一下进行计算的两个密文的logq和logp是否对的上
void test(Ciphertext a,Ciphertext b)
{
	if(a.logq != b.logq)
		cout<<"\nsome logq is not equal!!!!\n\n\n\n\n\n";
	if(a.logp!=b.logp || a.logp !=logp || b.logp!=logp)
		cout<<"\n"<<a.logp<<"  "<<b.logp<<"  "<<"some logp is wrong without reScale";
	return ;
}

//类的话初始化scheme一定要传引用，而且一定要用scheme:(scheme)的方式
//并且类的变量含有scheme的话，就不能够嵌套另一个有scheme的类
//就是说不能够     class Net(scheme)，然后Net中的成员变量包含complexconv1d
//此外，因为返回值都是void，所以会大量的传引用，这样子也可以节省内存。
//还有就是HEAAN的addConst没有复数的，只能addConstAndEqual
class ComplexConv1d
{
	public:
		ComplexConv1d(Scheme& scheme,int in_,int out_,int kernel_,int stride_ = 1,int padding_ = 0); //初始化
		void load_params(string filename,int i);													 //读取参数如bias和权重
		void forward(vector< vector<Ciphertext > > &data);											 //卷积层
		void complex_avg_pool1d(vector<vector<Ciphertext > >&data,int kernel=3,int stride=3,int padding = 0,int number=0);//池化层
		void complex_relu(vector<vector<Ciphertext> >& data);										 //激活函数

	public:
		int in_channels;
		int out_channels;
		int kernel_size;
		int stride;
		int padding;
		vector<complex<double> > bias;
		vector<vector<vector<complex<double> > > >channels;
		Scheme& scheme;
};


class ComplexLinear
{
	public:
		ComplexLinear(Scheme& scheme,int in_,int out_);
		void load_params(string filename,int i);
		void forward(vector<Ciphertext > &data);
		void complex_relu(vector<Ciphertext> & data);//同complexconv1d

	public:
		int in_;
		int out_;
		vector<complex<double> > bias;
		vector<vector<complex<double> > >weights;
		Scheme& scheme;
};
//两个类的构造函数，注意 : scheme(scheme)，还有就是要把权重和参数的大小初始化好
ComplexConv1d::ComplexConv1d(Scheme& scheme,int in_,int out_,int kernel_,int stride_,int padding_ ) : scheme(scheme)
{
	in_channels = in_;
	out_channels = out_;
	kernel_size = kernel_;
	stride = stride_;
	padding = padding_;
	vector<vector<complex<double> > >tmp;
	for (int i = 0; i < in_channels; ++i)
		tmp.push_back(vector<complex<double> >(kernel_size,complex<double>(0,0)));
	complex<double> temp123(0,0);
	for (int i = 0; i < out_channels; ++i)
	{
		channels.push_back(tmp);
		bias.push_back(temp123);
	}
}

ComplexLinear::ComplexLinear(Scheme& scheme,int in_,int out_) : scheme(scheme)
{
	this->in_ = in_;
	this->out_ = out_;
	for (int i = 0; i < out_; ++i)
	{	
		weights.push_back(vector<complex<double> >(in_,complex<double>(0,0)));
		bias.push_back(complex<double>(0,0));
	}
			
}
//读取权重和参数，注意读取的顺序要和保存的顺序对的上
void ComplexConv1d::load_params(string filename,int i)
{
	
	string rw,rb,iw,ib;
	rw = "conv"+to_string(i)+".conv_r.weight";
	rb = "conv"+to_string(i)+".conv_r.bias";
	iw = "conv"+to_string(i)+".conv_i.weight";
	ib = "conv"+to_string(i)+".conv_i.bias";
    ifstream rfile,ifile;
    rfile.open(filename + rw, ios::in);
    ifile.open(filename + iw, ios::in);
    for(int i = 0;i<channels.size();i++)
    {
    	for(int j =0;j<channels[0].size();j++)
    	{
    		for(int k =0;k<channels[0][0].size();k++)
    		{
    			double real,imag;
    			rfile>>real;
    			ifile>>imag;
    			if(real == imag &&real ==0 )
    				cout<<i<<" "<<j<<" "<<k<<endl;
    			channels[i][j][k].real(real);
    			channels[i][j][k].imag(imag);
    		}
    	}
    }
    rfile.close();
    ifile.close();
    rfile.open(filename + rb,ios::in);
    ifile.open(filename + ib,ios::in);
    for(int i =0;i<bias.size();i++)
    {
    	double real,imag;
    	rfile>>real;
    	ifile>>imag;
    	bias[i].real(real);
    	bias[i].imag(imag);
    }
    //别忘了close
    rfile.close();
    ifile.close();
}

void ComplexLinear::load_params(string filename,int i)
{
	
	string rw,rb,iw,ib;
	rw = "fc"+to_string(i)+".fc_r.weight";
	rb = "fc"+to_string(i)+".fc_r.bias";
	iw = "fc"+to_string(i)+".fc_i.weight";
	ib = "fc"+to_string(i)+".fc_i.bias";
    ifstream rfile,ifile;

    rfile.open(filename + rw, ios::in);
    ifile.open(filename + iw, ios::in);
    for(int i = 0;i<weights.size();i++)
    {
    	for(int j =0;j<weights[0].size();j++)
		{
			double real,imag;
			rfile>>real;
			ifile>>imag;
			weights[i][j].real(real);
			weights[i][j].imag(imag);
		}
    }
    rfile.close();
    ifile.close();
    rfile.open(filename + rb,ios::in);
    ifile.open(filename + ib,ios::in);
    for(int i =0;i<bias.size();i++)
    {
    	double real,imag;
    	rfile>>real;
    	ifile>>imag;
    	bias[i].real(real);
    	bias[i].imag(imag);
    }
    rfile.close();
    ifile.close();

}
//卷积层，只传进来一个data的引用，最后算出结果了就用data.swap(res)
//直接交换内存，用别的东西来取结果的话因为ciphertext并没有等号之类的
//会直接报告函数错误而不会说没有等号
//也就是说，涉及到ciphertext的操作尽量用内存来赋值
void ComplexConv1d::forward(vector< vector<Ciphertext> > &data )
{
	//初始化一个零的密文，因为后面很多addAndEqual这种，所以一个零的密文很重要
	Ciphertext czero;
	complex<double> qwe[n]; 
	for(int i =0;i<n;i++)
		qwe[i] = complex<double>(0,0);
	scheme.encrypt(czero, qwe, n, logp, logq);
	//padding一下 虽然在我们的网络中实际上没有用到
	for(int i = 0;i<in_channels;i++)//
	{
		for(int j = 0;j<padding;j++)
		{			
			data[i].insert(data[i].begin(),czero);
			data[i].insert(data[i].end(),czero);
		}
	}
	//计算出最后的输出结果的size
	int output_size = (data[0].size()-kernel_size +2*padding)/stride +1; 
	vector<vector<Ciphertext > > res;
	//开始卷积，对每一个out_channels，对卷积核覆盖的值，一口气遍历所有inchannels来得出值，然后再移动卷积模板，
	for(int i = 0; i<out_channels;i++)//
	{
		//temp_res存放一个输出channels的值，有关并行的东西需要注意数据冲突
		vector<Ciphertext > temp_res( output_size,czero );
		#pragma omp parallel for schedule(dynamic)
		for(int j = 0; j<=data[0].size()-kernel_size;j+=stride)//
		{
			for(int l = 0;l<in_channels;l++)//
			{
				Ciphertext ctemp2;
				scheme.encrypt(ctemp2, qwe, n, logp, logq);
				scheme.modDownToAndEqual(ctemp2, data[0][0].logq - logpc);
				for(int iter_kernel = 0;iter_kernel<kernel_size;iter_kernel++)//
				{
					Ciphertext temp_multi;
					scheme.multByConst(temp_multi,data[l][j+iter_kernel],channels[i][l][iter_kernel],logpc);
					scheme.reScaleByAndEqual(temp_multi,logpc);
					// test(ctemp2,temp_multi);
					scheme.addAndEqual(ctemp2, temp_multi);

				}
				scheme.modDownToAndEqual(temp_res[j], ctemp2.logq);
				// test(temp_res[j],ctemp2);
				scheme.addAndEqual(temp_res[j],ctemp2);//
			}
			//最终加上一个bias
			scheme.addConstAndEqual(temp_res[j],bias[i] + complex<double>(-1*bias[i].imag(),bias[i].real()),logpc);
		}
		//这个时候已经没有并行了，所以可以push，如果是在并行的大括号里面的话就不能push，只能够[j] = temp来赋值
		res.push_back(temp_res);
	}
	//can and only can swap  否则就再传一个引用过来。
	data.swap(res);
}

//跟卷积层的很像，但是要注意并行的数据冲突
void ComplexLinear::forward(vector<Ciphertext >& data)
{
	complex<double> qwe[n]; 
	for(int i =0;i<n;i++)
		qwe[i] = complex<double>(0,0);
	Ciphertext ctemp;
	scheme.encrypt(ctemp, qwe, n, logp, logq);
	//初始化好大小和零，不这么做容易数据冲突
	vector<Ciphertext > res(out_,ctemp);

	//这里并行的时候写入的对象都是res[i]，所以不会有数据冲突，不能够用res.push，并且相关的temp都要在里面初始化。
	#pragma omp parallel for schedule(dynamic)
	for(int i =0;i<out_;i++)
	{
		for(int j =0;j<in_;j++)
		{
			Ciphertext temp_multi;
			scheme.multByConst(temp_multi,data[j],weights[i][j],logpc);
			scheme.reScaleByAndEqual(temp_multi,logpc);
			scheme.modDownToAndEqual(res[i], temp_multi.logq);
			scheme.addAndEqual(res[i], temp_multi);
		}
		scheme.addConstAndEqual(res[i],bias[i] + complex<double>(-1*bias[i].imag(),bias[i].real()),logpc);
	}
	data.swap(res);
}

//把二维的data拍平成一维的输入线性层
void reshape(vector <vector<Ciphertext > >&data,vector<Ciphertext>& res)
{
	for(int i =0;i<data.size();i++)
	{
		for(int j =0;j<data[0].size();j++)
		{
			res.push_back(data[i][j]);
		}
	}
	//这里其实是在释放内存。
	vector <vector<Ciphertext > > data213;
	data.swap(data213);
}

//非常的简单
// 1.激活函数：
// def tri_deri_iso_fit_2_com(input_r,input_i):
//   return 6.40687926e-07*(tri(input_r)-3*square(input_i)*input_r)+1.05450276e-01*(square(input_r)-square(input_i))+0.5*input_r+0.5,\
//          6.40687926e-07*(square(input_r)*3*input_i-tri(input_i))+1.05450276e-01*(2*input_r*input_i)+0.5*input_i+0.5
void ComplexConv1d::complex_relu(vector<vector<Ciphertext > > &data)
{
	for(int i =0;i<data.size();i++)
	{
		#pragma omp parallel for schedule(dynamic)
		for(int j =0;j<data[0].size();j++)
		{

			Ciphertext x3,x2,res;
			scheme.mult(x2,data[i][j],data[i][j]);
			scheme.reScaleByAndEqual(x2,logp);

			scheme.modDownToAndEqual(data[i][j],x2.logq);
			scheme.mult(x3,data[i][j],x2);
			scheme.reScaleByAndEqual(x3,logp);

			scheme.multByConstAndEqual(x3,complex<double> (2.55477831e-07,0),logpc);
			scheme.reScaleByAndEqual(x3,logpc);

			scheme.multByConstAndEqual(x2,complex<double> (7.66023077e-02,0),logpc);
			scheme.reScaleByAndEqual(x2,logpc);

			scheme.multByConstAndEqual(data[i][j],complex<double>(0.5,0),logpc);
			scheme.reScaleByAndEqual(data[i][j],logpc);

			scheme.modDownToAndEqual(x2,x3.logq);
			scheme.modDownToAndEqual(data[i][j],x3.logq);

			scheme.addAndEqual(data[i][j],x3);
			scheme.addAndEqual(data[i][j],x2);
			scheme.addConstAndEqual(data[i][j],complex<double>(0.5,0.5),logpc);
		}
	}
	return ;
}
//比上面还简单
void ComplexLinear::complex_relu(vector<Ciphertext> & data)
{
	#pragma omp parallel for schedule(dynamic)
	for(int j =0;j<data.size();j++)
	{

		Ciphertext x3,x2,res;
		scheme.mult(x2,data[j],data[j]);
		scheme.reScaleByAndEqual(x2,logp);

		scheme.modDownToAndEqual(data[j],x2.logq);
		scheme.mult(x3,data[j],x2);
		scheme.reScaleByAndEqual(x3,logp);

		scheme.multByConstAndEqual(x3,complex<double> (2.55477831e-07,0),logpc);
		scheme.reScaleByAndEqual(x3,logpc);

		scheme.multByConstAndEqual(x2,complex<double> (7.66023077e-02,0),logpc);
		scheme.reScaleByAndEqual(x2,logpc);

		scheme.multByConstAndEqual(data[j],complex<double>(0.5,0),logpc);
		scheme.reScaleByAndEqual(data[j],logpc);

		scheme.modDownToAndEqual(x2,x3.logq);
		scheme.modDownToAndEqual(data[j],x3.logq);

		scheme.addAndEqual(data[j],x3);
		scheme.addAndEqual(data[j],x2);
		scheme.addConstAndEqual(data[j],complex<double>(0.5,0.5),logpc);
	}
	return ;
}

//池化层，我把并行放在了调用池化层的时候，其余都一样的
void ComplexConv1d::complex_avg_pool1d(vector<vector<Ciphertext > >  &data,int kernel,int stride,int padding,int number)
{
	Ciphertext czero;
	complex<double> qwe[n]; 
	for(int i =0;i<n;i++)
		qwe[i] = complex<double>(0,0);
	scheme.encrypt(czero, qwe, n, logp, logq);
	scheme.modDownToAndEqual(czero,data[0][0].logq);
	int length = data[0].size()/kernel;
	cout<<length<<"\n";

	if(number>=3)
	{
		#pragma omp parallel for schedule(dynamic)	
		for(int i =0;i<data.size();i++)
		{
			vector<Ciphertext > temp(length,czero);
			// #pragma omp parallel for schedule(dynamic)
			for(int j = 0;j<length;j++)
			{
				scheme.add(temp[j],temp[j],data[i][j*kernel]);
				scheme.add(temp[j],temp[j],data[i][j*kernel+1]);
				scheme.add(temp[j],temp[j],data[i][j*kernel+2]);
				scheme.multByConstAndEqual(temp[j],complex<double>(1.0/kernel,0),logpc);
				scheme.reScaleByAndEqual(temp[j],logpc);
		// 		// temp.push_back(temp4);
			}
			data[i].swap(temp);
		}
	}
	else
	{
		// #pragma omp parallel for schedule(dynamic)	
		for(int i =0;i<data.size();i++)
		{
			vector<Ciphertext > temp(length,czero);
			#pragma omp parallel for schedule(dynamic)
			for(int j = 0;j<length;j++)
			{
				scheme.add(temp[j],temp[j],data[i][j*kernel]);
				scheme.add(temp[j],temp[j],data[i][j*kernel+1]);
				scheme.add(temp[j],temp[j],data[i][j*kernel+2]);
				scheme.multByConstAndEqual(temp[j],complex<double>(1.0/kernel,0),logpc);
				scheme.reScaleByAndEqual(temp[j],logpc);
		// 		// temp.push_back(temp4);
			}
			data[i].swap(temp);
		}
	}
	// data.swap(temp);	
}


//释放vt的内存
void ClearVector( vector<Ciphertext>& vt ) 
{
    vector<Ciphertext>  vtTemp; 
    vtTemp.swap(vt);
}

int main()
{	
	TimeUtils timeutils;
	//读取数据
	cout<<logN<<endl;
	ifstream ifile;
    // ifile.open("/home/zenghuicong/HEAAN/run/data/all_wave.data", ios::in);
    ifile.open("/home/zenghuicong/HEAAN/run/data/x_val.txt", ios::in);
    vector<vector<complex<double> > >data;
    for(int i =0;i<4096;i++)
    {
    	vector<complex<double> > temp;
    	for(int j =0;j<8000;j++)
    	{
    		double a;
    		complex<double> b;
    		ifile >>a;
    		b.real(a);
    		b.imag(0);
    		temp.push_back(b);
    	}
    	data.push_back(temp);
    }
    //看下有没有读错
    cout<<"The first data"<<data[0][0]<<endl;
    ifile.close();


    ifile.open("/home/zenghuicong/HEAAN/run/data/y_val.txt", ios::in);
    vector<double >y;
    for(int i =0;i<4096;i++)
    {
    	double temp;
    	ifile >>temp;
    	y.push_back(temp);
    }
    //看下有没有读错
    cout<<"The first y "<<y[0]<<endl;
    ifile.close();
	timeutils.start("begin scheme");
    //初始化scheme
	srand(time(NULL));
	SetNumThreads(40);
	Ring ring;
	SecretKey secretKey(ring);
	Scheme scheme(secretKey, ring);
	//把数据加密到cipher中
	timeutils.stop("stop scheme");
	cout<<getCurrentRSS()<<"asdasdasd\n";

	for(int qq = 0;qq<4;qq++)
	{
		timeutils.start("begin enc");
		vector<Ciphertext> cipher;
		for(int i =0;i<data[0].size();i++)
		{
			complex<double>* plain = new complex<double>[n];
			for(int j =0;j<n;j++)
				plain[j] = data[j+qq*n][i];
			Ciphertext ctemp;
			scheme.encrypt(ctemp,plain,n,logp,logq);
			cipher.push_back(ctemp);
		}
		timeutils.stop("stop enc");
		cout<<getCurrentRSS()<<"asdasdasd\n";
		cout<<"\ncry done\n";
		//初始化类并且加载参数
		ComplexConv1d conv_list[6]= {
									ComplexConv1d(scheme,in_chanels[0],out_chanels[0],kernels[0]),
									ComplexConv1d(scheme,in_chanels[1],out_chanels[1],kernels[1]),
									ComplexConv1d(scheme,in_chanels[2],out_chanels[2],kernels[2]),
									ComplexConv1d(scheme,in_chanels[3],out_chanels[3],kernels[3]),
									ComplexConv1d(scheme,in_chanels[4],out_chanels[4],kernels[4]),
									ComplexConv1d(scheme,in_chanels[5],out_chanels[5],kernels[5])
									};

		ComplexLinear fc_list[2]={
									ComplexLinear(scheme,fc_in[0],fc_out[0]),
									ComplexLinear(scheme,fc_in[1],fc_out[1])
								};
		for (int i = 0; i < 6; ++i)
			conv_list[i].load_params("/home/zenghuicong/HEAAN/run/params/",i+1);
		for (int i = 0; i < 2; ++i)
			fc_list[i].load_params("/home/zenghuicong/HEAAN/run/params/",i+1);

		cout<<"Net done\n";
		//开始整活

		{
			cout<<"begin forward\n";
			vector<vector<Ciphertext > >res;
			res.push_back(cipher);
			cout<<"begin forward for\n";
			for (int i = 0; i < 6; ++i)
			{
				timeutils.start("convvvvvvvvvvvvvvvvvvvvvvvvvvvv");
				//开始卷积
				cout<<"begin conv\n";
				conv_list[i].forward(res);						
				//输出数据验证结果
				// complex<double>* res_temp = scheme.decrypt(secretKey,res[0][0]);
				// cout<<"data:"<<res_temp[0]<<"   "<<res[0][0].logq<<" \n";
				// cout<<"w:"<<conv_list[i].channels[0][0][0]<<"  "<<conv_list[i].bias[0]<<endl;
timeutils.stop("convvvvvvvvvvvvvvvvvvvvvvvvvvvv");
				timeutils.start("relu");
				//开始激活函数
				// cout<<"begin relu\n";
				conv_list[i].complex_relu(res);
				//输出数据验证结果
				// res_temp = scheme.decrypt(secretKey,res[0][0]);
				// cout<<"data:"<<res_temp[0]<<"   "<<res[0][0].logq<<" \n";

timeutils.stop("relu");
				// //开始池化
				// cout<<"begin pool\n";
				// cout<<"befor pool 's size:"<<res.size()<<" "<<res[0].size()<<endl;
				
				timeutils.start("pool");
				// #pragma omp parallel for schedule(dynamic)
				// for(int sb=0;sb<res.size();sb++)
				// {
				conv_list[i].complex_avg_pool1d(res,kernel,stride,0,i);//这里把并行拿出来是无奈之举，反正这样能跑通就对了
				// }	


				// cout<<"after pool 's size:"<<res.size()<<" "<<res[0].size()<<endl;		
				// res_temp = scheme.decrypt(secretKey,res[0][0]);
				// cout<<"data:"<<res_temp[0]<<" "<<res[0][0].logq<<" \n";;
				timeutils.stop("pool");

			}

			vector<Ciphertext >D1_res;
			complex<double>* res_temp ;
			// cout<<"befor reshape 's size:"<<res.size()<<" "<<res[0].size()<<" \n"; 
			reshape(res,D1_res);
			// cout<<"after reshape 's size:"<<D1_res.size()<<endl;

			//第一个线性层
			timeutils.start("fccccccccccccccccccccccccccccc1");
			fc_list[0].forward(D1_res);
			timeutils.stop("fccccccccccccccccccccccccccccc1");
			//输出结果看看
			// res_temp = scheme.decrypt(secretKey,D1_res[0]);
			// cout<<"fc 1:"<<res_temp[0]<<" logq:"<<D1_res[0].logq<<" \n";

			//第一个线性层的激活函数
			timeutils.start("fc accccccccccccccccccccccccccccct1");
			fc_list[0].complex_relu(D1_res);
			timeutils.stop("fc accccccccccccccccccccccccccccct1");
			//输出结果看看
			// res_temp = scheme.decrypt(secretKey,D1_res[0]);
			// cout<<"fact 1:"<<res_temp[0]<<" logq:"<<D1_res[0].logq<<" \n";

			//第二个线性层
			timeutils.start("fccccccccccccccccccccccccccccc2");
			fc_list[1].forward(D1_res);
			timeutils.stop("fccccccccccccccccccccccccccccc2");
			//输出最终结果
			// cout<<"The last logq:"<<D1_res[0].logq<<" \nThe final result";


			//计算命中率
			complex<double>* final_res[10] ;
			// vector<vector<complex<double> > > final()
			timeutils.start("dec");
			for(int sb =0;sb<10;sb++)
				final_res[sb] = scheme.decrypt(secretKey,D1_res[sb]);
			timeutils.stop("dec");
			int acc = 0;
			for(int i =0;i<n;i++)
			{
				double MAX = -999;
				int final_pos = 0;
				for(int j =0;j<10;j++)
				{
					double final_temp = final_res[j][i].real()*final_res[j][i].real() + final_res[j][i].imag()*final_res[j][i].imag(); 
					if(final_temp > MAX)
					{
						MAX = final_temp;
						final_pos = j;
					}
				}
				if(final_pos == y[i+qq*n])
					acc++;
			}
			cout<<endl;

		// timeutils.stop("stop eva");
		cout<<"acc : "<<acc <<" "<<double(acc/n)<<"\n";
		};
	}
	return 0;
}