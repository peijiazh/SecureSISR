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
// cnn fc Settings
int in_chanels[]={1,32,32,64,128,128};
int out_chanels[]={32,32,64,128,128,192};
int kernels[] ={13,11,9,7,5,3};
int fc_in[]={192*9,256};
int fc_out[]={256,10};
double dp_rate = 0.7;
//pooling setting
int stride = 3;
int kernel = 3;

long logq = 1200; ///< Ciphertext Modulus
long logp = 30; ///< Real message will be quantized by multiplying 2^40
long logn = 1; ///< log2(The number of slots)
long logpc = 30;
long n = 1 << logn;

void reshape(vector <vector<Ciphertext > >&data,vector<Ciphertext>& res);

void test(Ciphertext a,Ciphertext b)
{
	if(a.logq != b.logq)
		cout<<"\nsome logq is not equal!!!!\n\n\n\n\n\n";
	if(a.logp!=b.logp || a.logp !=logp || b.logp!=logp)
		cout<<"\n"<<a.logp<<"  "<<b.logp<<"  "<<"some logp is wrong without reScale";
	return ;
}

class ComplexConv1d
{
	public:
		//Ð´ºÃ·ÅÔÚÏÂÃæÁË
		ComplexConv1d(Scheme& scheme,int in_,int out_,int kernel_,int stride_ = 1,int padding_ = 0);
		//load ½ø ÄÇÐ© ²ÎÊýµ½channelsÀ´
		void load_params(string filename,int i);
		//1D ¾í»ý²Ù×÷
		void forward(vector< vector<Ciphertext > > &data,SecretKey secretKey);
		void complex_avg_pool1d(SecretKey & secretKey,vector<Ciphertext > &data,int kernel=3,int stride=3,int padding = 0);
		void complex_relu(vector<vector<Ciphertext> >& data);

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
		//Ð´ºÃ·ÅÔÚÏÂÃæÁË
		ComplexLinear(Scheme& scheme,int in_,int out_);

		//load ½ø ÄÇÐ© ²ÎÊýµ½Weights À´
		void load_params(string filename,int i);
		//È«Á¬½Ó
		void forward(vector<Ciphertext > &data);
		void complex_relu(vector<Ciphertext> & data);

	public:
		int in_;
		int out_;
		vector<complex<double> > bias;
		vector<vector<complex<double> > >weights;
		Scheme& scheme;
};

void ComplexConv1d::load_params(string filename,int i)
{
	
	string rw,rb,iw,ib;
	rw = "conv"+to_string(i)+".conv_r.weight";
	rb = "conv"+to_string(i)+".conv_r.bias";
	iw = "conv"+to_string(i)+".conv_i.weight";
	ib = "conv"+to_string(i)+".conv_i.bias";
    ifstream rfile,ifile;
    // vector<vector<vector<complex<double> > > >channels;
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
    // vector<complex<double> > bias;
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
// to_string
}

void ComplexLinear::load_params(string filename,int i)
{
	
	string rw,rb,iw,ib;
	rw = "fc"+to_string(i)+".fc_r.weight";
	rb = "fc"+to_string(i)+".fc_r.bias";
	iw = "fc"+to_string(i)+".fc_i.weight";
	ib = "fc"+to_string(i)+".fc_i.bias";
    ifstream rfile,ifile;

    // vector<vector<complex<double> > >weights;
	// vector<complex<double> > bias;
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
// to_string
}

ComplexConv1d::ComplexConv1d(Scheme& scheme,int in_,int out_,int kernel_,int stride_,int padding_ ) : scheme(scheme)
{
	in_channels = in_;
	out_channels = out_;
	kernel_size = kernel_;
	stride = stride_;
	padding = padding_;
	vector<vector<complex<double> > >tmp;
	//Ó¦¸ÃÓÐout_channels * in_channels *kernel_size¸öÂË²¨Æ÷£¬ÈýÎ¬vector£¬³õÊ¼»¯ÏÂ·ÖÅäÄÇÃ´¶à¸öÄÚ´æ
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
	//Ó¦¸ÃÓÐout_ * in_ ÄÇÃ´¶à¸öweight£¬2Î¬vector
	for (int i = 0; i < out_; ++i)
		weights.push_back(vector<complex<double> >(in_,complex<double>(0,0)));
			
}


void ComplexConv1d::forward(vector< vector<Ciphertext> > &data,SecretKey secretKey )
{
	
	Ciphertext czero;
	complex<double> qwe[n]; 
	for(int i =0;i<n;i++)
		qwe[i] = complex<double>(0,0);
	scheme.encrypt(czero, qwe, n, logp, logq);
	for(int i = 0;i<in_channels;i++)//Ç°ºópadding
	{
		for(int j = 0;j<padding;j++)
		{			
			data[i].insert(data[i].begin(),czero);
			data[i].insert(data[i].end(),czero);
		}
	}
	int output_size = (data[0].size()-kernel_size +2*padding)/stride +1; // Êä³öµÄµ¥¸öÊý¾Ý¶à³¤
	cout<<"\nreal begin conv\n";
	vector<vector<Ciphertext > > res;

	complex<double> * asd;
	
	for(int i = 0; i<out_channels;i++)//¶ÔÃ¿Ò»¸öout_channels£¬±éÀúËùÓÐin_channels
	{
		cout<<" outchannels is "<<i;
		vector<Ciphertext > temp_res( output_size,czero );
		#pragma omp parallel for schedule(dynamic)
		for(int j = 0; j<=data[0].size()-kernel_size;j+=stride)//ÐèÒª±éÀúµÄÊý¾ÝÊÇ:Êý¾Ý³¤¶È*in_channels£¬Ã¿´ÎÑ­»·µÃµ½Ò»¸ö¾í»ý½á¹û!!×¢ÒâÕâÊÇj+=stride
		{
			 // if((j%1000) == 0)
				// cout<<j<<" ";
			// Ciphertext ctemp;
			// scheme.encrypt(ctemp, qwe, n, logp, logq);
			for(int l = 0;l<in_channels;l++)//¶ÔËùÓÐµÄin_chanels½øÐÐ¾í»ýºóÇóºÍµÃµ½µÚÒ»¸ö¾í»ý½á¹û
			{
				Ciphertext ctemp2;
				scheme.encrypt(ctemp2, qwe, n, logp, logq);
				scheme.modDownToAndEqual(ctemp2, data[0][0].logq - logpc);
				for(int iter_kernel = 0;iter_kernel<kernel_size;iter_kernel++)//¾í£¬ÎÒ·è¿ñµÄ¾í
				{
					Ciphertext temp_multi;
					scheme.multByConst(temp_multi,data[l][j+iter_kernel],channels[i][l][iter_kernel],logpc);
					scheme.reScaleByAndEqual(temp_multi,logpc);
					test(ctemp2,temp_multi);
					scheme.addAndEqual(ctemp2, temp_multi);

					if(i ==0&&j==0&&l==0)
					{
						asd = scheme.decrypt(secretKey,ctemp2);
						cout<<"asd:"<<asd[0]<<" ";
						asd = scheme.decrypt(secretKey,temp_multi);
						cout<<"mult asd:"<<asd[0]<<" \n";
					}




				}
				scheme.modDownToAndEqual(temp_res[j], ctemp2.logq);
				test(temp_res[j],ctemp2);
				scheme.addAndEqual(temp_res[j],ctemp2);//Ã¿Ò»¸öin_channelsµÄ¾í»ýÖµÖ±½ÓÏà¼Ó¡£
				if(i ==0&&j==0)
				{
					asd = scheme.decrypt(secretKey,ctemp2);
					cout<<"a asd:"<<asd[0]<<" ";
					asd = scheme.decrypt(secretKey,temp_res[j]);
					cout<<"a mult asd:"<<asd[0]<<" \n";
				}
			}
			// qwe[0] = bias[i];
			if(i ==0&&j==0)
			{	cout<<"bias:"<<bias[i] + complex<double>(-1*bias[i].imag(),bias[i].real())<<endl;
				asd = scheme.decrypt(secretKey,temp_res[j]);
					cout<<"a last mult asd:"<<asd[0]<<" \n";
			}
			scheme.addConstAndEqual(temp_res[j],bias[i] + complex<double>(-1*bias[i].imag(),bias[i].real()),logpc);
			if(i ==0&&j==0)
			{	cout<<"bias2:"<<bias[i] + complex<double>(-1*bias[i].imag(),bias[i].real())<<endl;
				asd = scheme.decrypt(secretKey,temp_res[j]);
					cout<<"a last mult asd2:"<<asd[0]<<" \n";
			}
			// temp_res[j] = ctemp;
		}
		// scheme.addConstAndEqual(encIP2, degree3[1] / degree3[2], wBits - 2 * aBits);
		res.push_back(temp_res);
	}
	data.swap(res);
	cout<<endl;
}

void ComplexLinear::forward(vector<Ciphertext >& data)
{
	vector<Ciphertext > res;
	complex<double> qwe[n]; 
	for(int i =0;i<n;i++)
		qwe[i] = complex<double>(0,0);

	#pragma omp parallel for schedule(dynamic)
	for(int i =0;i<out_;i++)
	{
		Ciphertext ctemp;
		scheme.encrypt(ctemp, qwe, n, logp, logq);
		
		for(int j =0;j<in_;j++)
		{
			// temp += weights[i][j] * data[j];
			Ciphertext temp_multi;
			// qwe[0] = weights[i][j];
			scheme.multByConst(temp_multi,data[j],weights[i][j],logpc);
			scheme.reScaleByAndEqual(temp_multi,logpc);
			scheme.modDownToAndEqual(ctemp, temp_multi.logq);
			test(ctemp,temp_multi);
			scheme.addAndEqual(ctemp, temp_multi);
		}
		scheme.addConstAndEqual(ctemp,bias[i] + complex<double>(-1*bias[i].imag(),bias[i].real()),logpc);
		res.push_back(ctemp);   //complex<double>(-1*bias[i].imag(),bias[i].real());  不要pushback 直接[j] = ctemp
	}
	data.swap(res);
	// return res;
}

void reshape(vector <vector<Ciphertext > >&data,vector<Ciphertext>& res)
{
	for(int i =0;i<data.size();i++)
	{
		for(int j =0;j<data[0].size();j++)
		{
			res.push_back(data[i][j]);
		}
	}
	vector <vector<Ciphertext > > data213;
	data.swap(data213);
	cout<<"\n after free"<<data.size()<<endl;

}

void ComplexConv1d::complex_relu(vector<vector<Ciphertext > > &data)
{
	cout<<data.size()<<" ";
	cout<<data[0].size()<<endl;
	for(int i =0;i<data.size();i++)
	{
		// cout<<" 7987654321 ";
		// cout<<endl<<"5555"<<endl<<endl<<endl<<endl<<endl<<endl<<endl<<endl<<endl<<endl<<endl<<endl<<endl<<endl<<endl<<endl;
		// cout<<"qq";
		#pragma omp parallel for schedule(dynamic)
		for(int j =0;j<data[0].size();j++)
		{
			// if((j%1000) == 0)
			// 	cout<<"ww";
			Ciphertext x3,x2,res;
			scheme.mult(x2,data[i][j],data[i][j]);
			scheme.reScaleByAndEqual(x2,logp);
			// return complex<double> (2.55477831e-07,0) * tri(data) + 
			// complex<double> (7.66023077e-02,0) * square(data) + complex<double> (0.5,0)*data + complex<double> (0.5,0.5);///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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

void ComplexLinear::complex_relu(vector<Ciphertext> & data)
{
	cout<<data.size()<<" ";
		// cout<<" 7987654321 ";
		// cout<<endl<<"5555"<<endl<<endl<<endl<<endl<<endl<<endl<<endl<<endl<<endl<<endl<<endl<<endl<<endl<<endl<<endl<<endl;
		// cout<<"qq";
	#pragma omp parallel for schedule(dynamic)
	for(int j =0;j<data.size();j++)
	{
		// if((j%1000) == 0)
		// 	cout<<"e";
		Ciphertext x3,x2,res;
		scheme.mult(x2,data[j],data[j]);
		scheme.reScaleByAndEqual(x2,logp);
		// return complex<double> (2.55477831e-07,0) * tri(data) + 
		// complex<double> (7.66023077e-02,0) * square(data) + complex<double> (0.5,0)*data + complex<double> (0.5,0.5);///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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

//³Ø»¯²ã
void ComplexConv1d::complex_avg_pool1d(SecretKey & secretKey,vector<Ciphertext >  &data,int kernel,int stride,int padding)
{
	// cout<<"q";
	Ciphertext czero;
	complex<double> qwe[n]; 
	for(int i =0;i<n;i++)
		qwe[i] = complex<double>(0,0);
	// cout<<"w";
	scheme.encrypt(czero, qwe, n, logp, logq);
	// cout<<"e";
	complex<double>* asd;




	vector<vector<Ciphertext > > res;
	// #pragma omp parallel for schedule(dynamic)

		vector<Ciphertext > temp;
		
		// #pragma omp parallel for schedule(dynamic)
		for(int j = 0;j<=data.size()-kernel;j+=stride)
		{
			Ciphertext temp2,temp3,temp4;
			scheme.encrypt(temp2,qwe, n,logp,logq);
			scheme.modDownToAndEqual(temp2,data[j].logq);
			scheme.encrypt(temp3,qwe, n,logp,logq);
			scheme.modDownToAndEqual(temp3,data[j].logq);
			scheme.encrypt(temp4,qwe, n,logp,logq);
			scheme.modDownToAndEqual(temp4,data[j].logq);
			// if(j==0)
			// {	asd = scheme.decrypt(secretKey,temp2);cout<<"\nbefor asd:"<<asd[0]<<endl;}
			// #pragma omp parallel for schedule(dynamic)
			// for(int iter_kernel = 0;iter_kernel<kernel;iter_kernel++)
			// {
			// 	test(temp2,data[i][j+iter_kernel]);
			// 	scheme.addAndEqual(temp2,data[i][j+iter_kernel]);
			// 	if(j==0&&i==0)
			// 	{	asd = scheme.decrypt(secretKey,temp2);cout<<"\nadd asd:"<<asd[0]<<endl;
			// 		asd = scheme.decrypt(secretKey,data[i][j+iter_kernel]);cout<<"\nadd data asd:"<<asd[0]<<endl;}
			// }
			// 
			scheme.add(temp2,temp2,data[j]);
				// if(j==0)
				// {	asd = scheme.decrypt(secretKey,temp2);cout<<"\nadd asd:"<<asd[0]<<endl;
				// 	asd = scheme.decrypt(secretKey,data[j+0]);cout<<"\nadd datad:"<<asd[0]<<endl;}
			scheme.add(temp3,temp2,data[j+1]);
			// if(j==0)
			// 	{	asd = scheme.decrypt(secretKey,temp3);cout<<"\nadd asd:"<<asd[0]<<endl;
			// 		asd = scheme.decrypt(secretKey,data[j+1]);cout<<"\nadd datad:"<<asd[0]<<endl;}
			scheme.add(temp4,temp3,data[j+2]);
			// if(j==0)
			// 	{	asd = scheme.decrypt(secretKey,temp4);cout<<"\nadd asd:"<<asd[0]<<endl;
			// 		asd = scheme.decrypt(secretKey,data[j+2]);cout<<"\nadd datad:"<<asd[0]<<endl;}



			// if(j==0)
			// {	asd = scheme.decrypt(secretKey,temp4);cout<<"\nasd:"<<asd[0]<<endl;}
			scheme.multByConstAndEqual(temp4,complex<double>(1.0/kernel,0),logpc);
			scheme.reScaleByAndEqual(temp4,logpc);
			// if(j==0)
			// {	asd = scheme.decrypt(secretKey,temp4);cout<<"\nasd:"<<asd[0]<<endl;}
			// temp4 /= kernel;
			temp.push_back(temp4);
		}
	data.swap(temp);	
}


int main()
{	
	ifstream ifile;
    ifile.open("/home/zenghuicong/HEAAN/run/data/all_wave.data", ios::in);
    vector<vector<complex<double> > >data;
    for(int i =0;i<1;i++)
    {
    	vector<complex<double> > temp;
    	for(int j =0;j<8000;j++)
    	{
    		double a;
    		complex<double> b;
    		ifile >>a;
    		//cout<<a<<" ";
    		b.real(a);
    		b.imag(0);
    		temp.push_back(b);
    	}
    	data.push_back(temp);
    }
    cout<<data[0][0]<<endl;
    ifile.close();


	srand(time(NULL));
	SetNumThreads(40);
	TimeUtils timeutils;
	Ring ring;
	SecretKey secretKey(ring);
	Scheme scheme(secretKey, ring);

	vector<Ciphertext> cipher;
	// #pragma omp parallel for schedule(dynamic)
	for(int i =0;i<data[0].size();i++)
	{
		complex<double>* plain = new complex<double>[n];
		plain[0] = data[0][i];
		Ciphertext ctemp;
		scheme.encrypt(ctemp,plain,n,logp,logq);
		cipher.push_back(ctemp);
		// cout<<i<<" ";
	}
	cout<<"\ncry 231done\n";

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
	vector<Ciphertext > cres;
	{
		cout<<"begin forward\n";
		vector<vector<Ciphertext > >res;
		res.push_back(cipher);
		cout<<"begin forward for\n";
		//CNN ²ã´Î
		// res ÊÇ¶þÎ¬Êý¾Ý£¬channels * 1DÊý¾Ý£¬µÚÒ»²ãÊÇ1*8000ÊäÈë£¬32*´óÔ¼2300Êä³ö£¬È»ºóÏÂÒ»²ã32*XXX.....
		// for (int i = 0; i < 6; ++i)
		// {
		// 	// vector<vector<Ciphertext > > relu_res;
		// 	cout<<"begin conv\n";
				
		// 	conv_list[i].forward(res,secretKey);
		// 	cout<<"w:"<<conv_list[i].channels[0][0][0]<<"  "<<conv_list[i].bias[0]<<endl;
		// 	complex<double>* res_temp = scheme.decrypt(secretKey,res[i][0]);
		// 	cout<<"data:"<<res_temp[0]<<"   "<<res[0][0].logq<<" ";


		// 	cout<<"begin relu\n";
		// 	conv_list[i].complex_relu(res);
		// 	res_temp = scheme.decrypt(secretKey,res[i][0]);
		// 	cout<<"data:"<<res_temp[0]<<"   "<<res[0][0].logq<<" ";



		// 	cout<<"begin pool\n";
		// 	cout<<"hh:"<<res.size()<<res[0].size()<<endl;
		// 	#pragma omp parallel for schedule(dynamic)
		// 	for(int sb=0;sb<res.size();sb++)
		// 	{
		// 		conv_list[i].complex_avg_pool1d(secretKey,res[sb],kernel,stride,0);
		// 	}	

		// 	cout<<"\ngg:"<<res.size()<<res[0].size()<<endl;
		// 	for(int oo = 0;oo<9;oo++)
		// 	{		
		// 		res_temp = scheme.decrypt(secretKey,res[i][oo]);
		// 		cout<<"data:"<<res_temp[0]<<" ";
		// 	}
		// 	// cout<<res[0][0].logq<<" "<<i<<endl;
		// }
		cout<<res.size()<<" q "<<res[0].size()<<" q"; 
		vector<Ciphertext >D1_res;
		reshape(res,D1_res);
		// cout<<D1_res.size(); 
		//FC²ã
		complex<double>* res_temp ;
		cout<<endl<<D1_res.size()<<endl;
		fc_list[0].forward(D1_res);
		res_temp = scheme.decrypt(secretKey,D1_res[0]);
		cout<<res_temp[0]<<" 1 fc \n";

		fc_list[0].complex_relu(D1_res);
		res_temp = scheme.decrypt(secretKey,D1_res[0]);
		cout<<res_temp[0]<<" 2 fc \n";
		cout<<D1_res[0].logq<<" 1 fc active \n";

		fc_list[1].forward(D1_res);
		res_temp = scheme.decrypt(secretKey,D1_res[0]);
		cout<<res_temp[0]<<" 3 fc \n";
		cout<<D1_res[0].logq<<" 2 fc output \n";
		
	};
	// tempnet.forward(cipher);
	// for(int i =0;i<cres.size();i++)
	// {
	// 	complex<double>* res_temp = scheme.decrypt(secretKey,cres[i]);
	// 	cout<<res_temp[0]<<" ";
	// }
	return 0;
}