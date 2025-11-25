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

#include <bits/stdc++.h>
#include <omp.h>
using namespace std;
// 网络结构的参数
int in_chanels[]= {1,40,40,40,40,40,40,40,40};
int out_chanels[]={40,40,40,40,40,40,40,40,1};
int row_kernels_size[] ={3,3,3,3,3,3,3,3,3};
int col_kernels_size[] ={3,3,3,3,3,3,3,3,3};
double dp_rate = 0.7;
//pooling setting
int stride = 1;
int kernel = 3;

int size_row = 30;
int size_col = 30;
int test_sq = 10;

//加密方案的参数
long logq = 1200; ///< Ciphertext Modulus
long logp = 30; ///< Real message will be quantized by multiplying 2^40
long logn = 9; ///< log2(The number of slots)
long logpc = 30;
long n = 1 << logn;



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
class Conv2d
{
	public:
		Conv2d(Scheme& scheme,int in_,int out_,int row_kernel_,int col_kernel_,int stride_ = 1,int padding_ = 1); //初始化
		void load_params(string filename,int i);													 //读取参数如bias和权重
		void forward(vector<vector< vector<Ciphertext> > >&data);											 //卷积层
		void activ(vector<Ciphertext > &data);										 //激活函数

	public:
		int in_channels;
		int out_channels;
		int row_kernel_size;
		int col_kernel_size;
		int stride;
		int padding;
		vector<vector<vector<vector<complex<double> > > > >channels;
		Scheme& scheme;
};


//两个类的构造函数，注意 : scheme(scheme)，还有就是要把权重和参数的大小初始化好
Conv2d::Conv2d(Scheme& scheme,int in_,int out_,int row_kernel_,int col_kernel_,int stride_,int padding_ ) : scheme(scheme)
{
	in_channels = in_;
	out_channels = out_;
	row_kernel_size = row_kernel_;
	col_kernel_size = col_kernel_;
	stride = stride_;
	padding = padding_;

	// vector<vector<vector<vector<complex<double> > > > >channels;
	vector<vector<vector<complex<double> > > >tmp;
	for (int i = 0; i < in_channels; ++i)
		tmp.push_back(vector<vector<complex<double> > >(row_kernel_size,vector<complex<double> >(col_kernel_size,complex<double>(0,0))));
	for (int i = 0; i < out_channels; ++i)
	{
		channels.push_back(tmp);
	}
}

//读取权重和参数，注意读取的顺序要和保存的顺序对的上
void Conv2d::load_params(string filename,int i)
{
	string r,rb,iw,ib;
	r = to_string(i)+".weight";
    ifstream rfile,ifile;
    rfile.open(filename + r, ios::in);
    double real;
    for(int i = 0;i<channels.size();i++)
    {
    	for(int j =0;j<channels[0].size();j++)
    	{
    		for(int k =0;k<channels[0][0].size();k++)
    		{
    			for(int l=0;l<channels[0][0][0].size();l++)
    			{

	    			rfile>>real;
	    			// if(real == imag &&real ==0 )
	    			// 	cout<<i<<" "<<j<<" "<<k<<endl;
	    			channels[i][j][k][l].real(real);
	    			channels[i][j][k][l].imag(0);
    			}
    		}
    	}
    }
    rfile.close();
}


//卷积层，只传进来一个data的引用，最后算出结果了就用data.swap(res)
//直接交换内存，用别的东西来取结果的话因为ciphertext并没有等号之类的
//会直接报告函数错误而不会说没有等号
//也就是说，涉及到ciphertext的操作尽量用内存来赋值
void Conv2d::forward(vector<vector< vector<Ciphertext> > >&data )
{
	//初始化一个零的密文，因为后面很多addAndEqual这种，所以一个零的密文很重要
	Ciphertext czero;
	complex<double> qwe[n]; 
	for(int i =0;i<n;i++)
		qwe[i] = complex<double>(0,0);
	scheme.encrypt(czero, qwe, n, logp, logq);
	if(in_channels <out_channels)
	{
		// cout<<"qwe";
		for(int i =0;i<out_channels- in_channels;i++)
		{
			vector<vector<Ciphertext > > beiguoxia(size_row,vector<Ciphertext>(size_col,czero));
			data.push_back(beiguoxia);
		}
	}

	// cout<<"logq:"<<data[0][3][3].logq<<" "<<data[1][0][0].logq;
	vector<vector<vector<Ciphertext > > >res(out_channels,vector<vector<Ciphertext > >(2,vector<Ciphertext>(size_col,czero)));
	//开始卷积，对每一个out_channels，对卷积核覆盖的值，一口气遍历所有inchannels来得出值，然后再移动卷积模板，
	// #pragma omp parallel for schedule(dynamic)
	
	// cout<<"total logq:"<<data[0][3][3].logq<<" "<<data[1][0][0].logq<<endl;
	for(int j = 0;j<test_sq ;j+=stride)
	{
		#pragma omp parallel for schedule(dynamic)
		for(int i = 0; i<out_channels;i++)
		{
			if(i==1&&in_channels == 1)
				cout<<"logq:"<<data[0][3][3].logq<<" "<<data[1][0][0].logq<<endl;
			if(j>=2)
			{
				// cout<<"1";
				std::swap(data[i][j-2],res[i][0]);
				// scheme.addAndEqual
				// cout<<"2";
				activ(data[i][j-2]);
				// cout<<"3";
				std::swap(res[i][0],res[i][1]);
				// cout<<"4";
				vector<Ciphertext> temp_zero(size_col,czero);
				std::swap(res[i][1],temp_zero);
				// cout<<"5";
			}
			for(int k = 0;k<test_sq ;k+=stride)
			{
				// cout<<i<<" "<<j<<" "<<k<<endl;
				for(int l = 0;l<in_channels;l++)
				{
					// cout<<j<<endl;
					int row=0,col=0;
					int temp_row_kernel_size=row_kernel_size;
					int temp_col_kernel_size=col_kernel_size;
					Ciphertext ctemp2;
					scheme.encrypt(ctemp2, qwe, n, logp, logq);
					
					if(j==0)
						row++;
					else if(j==size_row-1)
						temp_row_kernel_size =  row_kernel_size-1;
					if(k==0)
						col++;
					else if(k==size_col-1)
						temp_col_kernel_size =  col_kernel_size-1;

					for(int x = row;x<temp_row_kernel_size;x++)
					{
						for(int y = col;y<temp_col_kernel_size;y++)
						{
							Ciphertext temp_multi;
							scheme.multByConst(temp_multi,data[l][j+x-1][k+y-1],channels[i][l][x][y],logpc);
							scheme.reScaleByAndEqual(temp_multi,logpc);
							// test(ctemp2,temp_multi);
							scheme.modDownToAndEqual(ctemp2, temp_multi.logq );
							scheme.addAndEqual(ctemp2, temp_multi);
						}
					}
					scheme.modDownToAndEqual(res[i][bool(j)][k], ctemp2.logq);
					// test(temp_res[j],ctemp2);
					scheme.addAndEqual(res[i][bool(j)][k],ctemp2);//

				}
			}
		}
	}
	#pragma omp parallel for schedule(dynamic)
	for(int i = 0; i<out_channels;i++)
	{
		std::swap(data[i][test_sq -2],res[i][0]);
		std::swap(data[i][test_sq -1],res[i][1]);
		activ(data[i][test_sq -2]);
		activ(data[i][test_sq -1]);
	}
	//can and only can swap  否则就再传一个引用过来。
}

//非常的简单
void Conv2d::activ(vector<Ciphertext > &data)
{

	if(out_channels == 1)
		return;
	#pragma omp parallel for schedule(dynamic)
	for(int j =0;j<data.size();j++)
	{

		Ciphertext x3,x2,res;
		scheme.mult(x2,data[j],data[j]);
		scheme.reScaleByAndEqual(x2,logp);

		scheme.modDownToAndEqual(data[j],x2.logq);
		scheme.mult(x3,data[j],x2);
		scheme.reScaleByAndEqual(x3,logp);

		scheme.multByConstAndEqual(x3,complex<double> (7.81250049e-05,0),logpc);
		scheme.reScaleByAndEqual(x3,logpc);

		scheme.multByConstAndEqual(x2,complex<double> (1.87500082e-01,0),logpc);
		scheme.reScaleByAndEqual(x2,logpc);

		scheme.multByConstAndEqual(data[j],complex<double>(4.99625000e-01,0),logpc);
		scheme.reScaleByAndEqual(data[j],logpc);

		scheme.modDownToAndEqual(x2,x3.logq);
		scheme.modDownToAndEqual(data[j],x3.logq);

		scheme.addAndEqual(data[j],x3);
		scheme.addAndEqual(data[j],x2);
		scheme.addConstAndEqual(data[j],complex<double>(0.3,0.0),logpc);
	}
	//7.81250049e-05*(tri(x))+1.87500082e-01*(square(x))+4.99625000e-01*x+0.3
	return ;
	// for(int l=0;l<data.size();l++)
	// {
	// 	for(int i =0;i<data[0].size();i++)
	// 	{
	// 		#pragma omp parallel for schedule(dynamic)
	// 		for(int j =0;j<data[0][0].size();j++)
	// 		{

	// 			Ciphertext x3,x2,res;
	// 			scheme.mult(x2,data[l][i][j],data[l][i][j]);
	// 			scheme.reScaleByAndEqual(x2,logp);

	// 			scheme.modDownToAndEqual(data[l][i][j],x2.logq);
	// 			scheme.mult(x3,data[l][i][j],x2);
	// 			scheme.reScaleByAndEqual(x3,logp);

	// 			scheme.multByConstAndEqual(x3,complex<double> (2.55477831e-07,0),logpc);
	// 			scheme.reScaleByAndEqual(x3,logpc);

	// 			scheme.multByConstAndEqual(x2,complex<double> (7.66023077e-02,0),logpc);
	// 			scheme.reScaleByAndEqual(x2,logpc);

	// 			scheme.multByConstAndEqual(data[l][i][j],complex<double>(0.5,0),logpc);
	// 			scheme.reScaleByAndEqual(data[l][i][j],logpc);

	// 			scheme.modDownToAndEqual(x2,x3.logq);
	// 			scheme.modDownToAndEqual(data[l][i][j],x3.logq);

	// 			scheme.addAndEqual(data[l][i][j],x3);
	// 			scheme.addAndEqual(data[l][i][j],x2);
	// 			scheme.addConstAndEqual(data[l][i][j],complex<double>(0.5,0.5),logpc);
	// 		}
	// 	}
	// }
	// return ;
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
	ifstream ifile;
	cout<<"logN:"<<logN<<" n:"<<n<<endl;
	int i=0,j=0;
    ifile.open("/home/zenghuicong/HEAAN/run/input.txt", ios::in);
    ////先只读取一副图像试试水
    vector<vector<complex<double> > >data(size_row,vector<complex<double> >(size_col,complex<double>(0,0)));
    // cout<<data.size()<<data[0].size();
    double a;
    for(i =0;i<size_row;i++)
    {
    	for(j =0;j<size_col;j++)
    	{
    		ifile>>a;
    		data[i][j].real(a);
    		// data[i][j].imag(0);    		
    	}
    }
    ifile.close();
    //看下有没有读错
    // 
    cout<<"The first data"<<data[0][0]<<endl;;

    
    //初始化scheme
    timeutils.start("begin scheme");
	srand(time(NULL));
	SetNumThreads(40);
	Ring ring;
	SecretKey secretKey(ring);
	Scheme scheme(secretKey, ring);
	timeutils.stop(" stop shceme");
	//把数据加密到cipher中
	timeutils.start("begin encrypt");
	
	data[0][0].real(1.2345);
	data[0][1].real(2.3456);

	// Ciphertext czero;
	// complex<double> qwe[n]; 
	// for(int i =0;i<n;i++)
	// 	qwe[i] = complex<double>(0,0);
	// scheme.encrypt(czero, qwe, n, logp, logq);
	vector<vector<vector<Ciphertext > > >res(1,vector<vector<Ciphertext> > (size_row,vector<Ciphertext>(size_col)));
	

	for(i = 0;i<data.size();i++)
	{
		// cout<<i;
		#pragma omp parallel for schedule(dynamic)
		for(j = 0;j<data[0].size();j++)
		{
			complex<double>* plain = new complex<double>[n];
			plain[0] = data[i][j];
			// Ciphertext ctemp;
			scheme.encrypt(res[0][i][j],plain,n,logp,logq);
			// cipher[i][j].push_back(ctemp);	
		}	
	}
	timeutils.stop("stop encrypt");
	ofstream out("/home/zenghuicong/HEAAN/run/input.txt/output.txt",ios::trunc);

	// out<<res[0][0][0];
	// out<<res[0][0][1];
	ZZ abc = 1999999999999;
	cout<<abc;
	out.close();
	scheme.addAndEqual(res[0][0][0],res[0][0][1]);
	complex<double>* res_temp = scheme.decrypt(secretKey,res[0][0][0]);
	cout<<"data:"<<res_temp[0]<<"   "<<res[0][0][0].logq<<" \n";

	// cout<<"\nencrypt done\n";
	//初始化类并且加载参数

	return 0;
}