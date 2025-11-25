#include <iostream>
#include <stdio.h>
#include <vector>
#include <sys/time.h>
#include <chrono>
#include <bits/stdc++.h>

#include <NTL/RR.h>
#include <NTL/xdouble.h>
#include <NTL/ZZ.h>
#include <NTL/ZZX.h>
#include "NTL/RR.h"
#include "NTL/vec_RR.h"
#include "NTL/mat_RR.h"
#include <NTL/BasicThreadPool.h>

#include <NTL/mat_ZZ.h>
#include <NTL/mat_poly_ZZ.h>
#include <NTL/ZZXFactoring.h>

#include "HEAAN/src/HEAAN.h"
#include <unistd.h>
#include <sys/resource.h>
#include <bits/stdc++.h>
#include <omp.h>
#include <cmath>
#include <fstream>
#include "settings.cpp"
typedef complex<double> cdouble;

using namespace NTL;
using namespace std;

#ifndef __MYUTILS_CPP__
#define __MYUTILS_CPP__

// Return instantaneous process memory usage
double getCurrentRSS()
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
    return double((size_t)rss * (size_t)sysconf( _SC_PAGESIZE))/1024/1024/1024;
}

// Return the smallest power of 2 greater than a given value
int getLeastLargerSecondPower(int num){
    int tmp = (int)log2(num);
    if(pow(2,tmp)==num){
        return tmp;
    }
    return tmp + 1;
}

// If parameters are incorrect, output warning and exit program
void warningAndExit(){
    cout << "Command error: insufficient parameters\n";
    exit(-1);
}

// Read image from specified path
vector<vector<double> > getMatrixFromPath(char * path,int sizeRow,int sizeCol){
    fstream file;
    file.open(path,ios::in|ios::out);

    vector<vector<double> >tmp(sizeRow,vector<double>(sizeCol,0));

    for (int i = 0; i < sizeRow; i++)
    {
        for (int j = 0; j < sizeCol; j++)
        {
            file >> tmp[i][j];
        }
        
    }

    file.close();
    return tmp;
} 

// Display an image
void plaintextDisplay(vector<vector<double> >&image){
    for (int i = 0; i < image.size(); i++)
    {
        for (int j = 0; j < image[0].size(); j++)
        {
            cout << setprecision(19) <<image[i][j] << " " ;
        }cout << "\n";
        
    }
}

// Display a complex image
void plaintextDisplay(vector<vector<cdouble> >&image){
    for (int i = 0; i < image.size(); i++)
    {
        for (int j = 0; j < image[0].size(); j++)
        {
            cout << setprecision(19) <<image[i][j] << " " ;
        }cout << "\n";
        
    }
}

// Convert floating-point matrix to complex matrix
vector<vector<cdouble> >  doubleToComplex(vector<vector<double> >&image){
    vector<vector<cdouble> > tmp (image.size(),vector<cdouble>(image[0].size(),0));
    for (int i = 0; i < image.size(); i++)
    {
        for (int j = 0; j < image[0].size(); j++)
        {
            tmp[i][j] = cdouble(image[i][j],0);
        }
    }
    
    return tmp;
}

// Helper for genWeightMasks
bool check(int x,int y,int sizeRow,int sizeCol){
    return x >= 1 && x <= sizeRow && y >= 1 && y <= sizeCol;
}

// Generate packing mask
vector<vector<cdouble > > genWeightMasks(int slots,int sizeRow,int sizeCol){
    // vector<vector<cdouble> >res(9,vector<cdouble>(slots,cdouble(0,0)));
    vector<vector<cdouble> >res(9,vector<cdouble>(slots*cipherNum,cdouble(0,0)));
    vector<vector<int> >direction{  {-1,-1},{-1,0},{-1,1},
                                    {0,-1},{0,0},{0,1},
                                    {1,-1},{1,0},{1,1},
                                };

    // Image range is between 1-sizeRow, 1-sizeCol, beyond that fill with 0
    for(int i=0;i<9;i++){
        for(int x = 1;x<=sizeRow;x++){
            for (int y = 1; y <= sizeCol; y++)
            {
                int tmpX = x + direction[i][0];
                int tmpY = y + direction[i][1];
                res[i][(x-1)*sizeCol+(y-1)] = cdouble(check(tmpX,tmpY,sizeRow,sizeCol) * 1,0);
                
            }
            
        }
    }

    return res;
    
}

// Generate mask for fast upsampling layer
vector<vector<cdouble > > genBicWeightMaska(int slots,int sizeRow,int sizeCol){
    // vector<vector<cdouble> >res(64,vector<cdouble>(slots,cdouble(0,0)));
    vector<vector<cdouble> >res(16*fa*fa,vector<cdouble>(slots*cipherNum,cdouble(0,0)));
    // vector<vector<int> >direction{{-3, -3}, {-3, -2}, {-3, -1},{-3, 0}, {-3, 1}, {-3, 2},{-3, 3}, {-3, 4},
    //              {-2, -3}, {-2, -2}, {-2, -1},{-2, 0}, {-2, 1}, {-2, 2},{-2, 3}, {-2, 4},
    //              {-1, -3}, {-1, -2}, {-1, -1},{-1, 0}, {-1, 1}, {-1, 2},{-1, 3}, {-1, 4},
    //              {0, -3}, {0, -2}, {0, -1},{0, 0}, {0, 1}, {0, 2},{0, 3},{0, 4},
    //              {1, -3}, {1, -2}, {1, -1},{1, 0}, {1, 1}, {1, 2},{1, 3}, {1, 4},
    //              {2, -3}, {2, -2}, {2, -1},{2, 0}, {2, 1}, {2, 2},{2, 3}, {2, 4},
    //              {3, -3}, {3, -2}, {3, -1},{3, 0}, {3, 1}, {3, 2},{3, 3}, {3, 4},
    //              {4, -3}, {4, -2},{4, -1},{4, 0}, {4, 1}, {4, 2},{4, 3}, {4, 4},
    // };

    vector<vector<int> >direction(16*fa*fa, vector<int>(2,0));
    int count = 0;
    for(int g=-(2*fa-1);g<=2*fa;g++){
        for(int h=-(2*fa-1);h<=2*fa;h++){
            direction[count][0] = g;
            direction[count][1] = h;
            count++;
        }
    }

    for(int i=0;i<16*fa*fa;i++){
        for(int x = 1;x<=sizeRow;x++){
            for (int y = 1; y <= sizeCol; y++)
            {
                int tmpX = x + direction[i][0];
                int tmpY = y + direction[i][1];
                res[i][(x-1)*sizeCol+(y-1)] = cdouble(check(tmpX,tmpY,sizeRow,sizeCol) * 1,0);
                
            }
            
        }
    }
    return res;
}

// Generate multi-ciphertext version of packing mask
vector<vector<vector<cdouble>>>  multigenWeightMasks(int slots,int sizeRow,int sizeCol, int cipherNum){
    vector<vector<cdouble> >res(9,vector<cdouble>(slots*cipherNum,cdouble(0,0)));
    vector<vector<int> >direction{  {-1,-1},{-1,0},{-1,1},
                                    {0,-1},{0,0},{0,1},
                                    {1,-1},{1,0},{1,1},
                                };
    // cout << slots*cipherNum << endl;
    for(int i=0;i<9;i++){
        for(int x = 1;x<=sizeRow;x++){
            for (int y = 1; y <= sizeCol; y++)
            {
                int tmpX = x + direction[i][0];
                int tmpY = y + direction[i][1];
                res[i][(x-1)*sizeCol+(y-1)] = cdouble(check(tmpX,tmpY,sizeRow,sizeCol) * 1,0);
                
            }
            
        }
    }

    vector<vector<vector<cdouble>>> tmp(9, vector<vector<cdouble>>(cipherNum, vector<cdouble>(slots, cdouble(0,0))));
    for(int i=0;i<9;i++){
        for(int x = 0; x<cipherNum-1;x++){
            for (int y = 0; y < slots; y++){
                tmp[i][x][y] = res[i][x*slots+y];
            }
        }
        for(int j=(cipherNum-1)*slots; j<sizeRow*sizeCol; j++){
            tmp[i][cipherNum-1][j-(cipherNum-1)*slots] = res[i][j];
        }
    }

    return tmp;
    
}

// Generate multi-ciphertext version of fast upsampling layer mask
vector<vector<vector<cdouble>>> multigenBicWeightMaska(int slots,int sizeRow,int sizeCol, int cipherNum){
    vector<vector<cdouble> >res(16*fa*fa,vector<cdouble>(slots*cipherNum,cdouble(0,0)));
    // vector<vector<int> >direction{{-3, -3}, {-3, -2}, {-3, -1},{-3, 0}, {-3, 1}, {-3, 2},{-3, 3}, {-3, 4},
    //              {-2, -3}, {-2, -2}, {-2, -1},{-2, 0}, {-2, 1}, {-2, 2},{-2, 3}, {-2, 4},
    //              {-1, -3}, {-1, -2}, {-1, -1},{-1, 0}, {-1, 1}, {-1, 2},{-1, 3}, {-1, 4},
    //              {0, -3}, {0, -2}, {0, -1},{0, 0}, {0, 1}, {0, 2},{0, 3},{0, 4},
    //              {1, -3}, {1, -2}, {1, -1},{1, 0}, {1, 1}, {1, 2},{1, 3}, {1, 4},
    //              {2, -3}, {2, -2}, {2, -1},{2, 0}, {2, 1}, {2, 2},{2, 3}, {2, 4},
    //              {3, -3}, {3, -2}, {3, -1},{3, 0}, {3, 1}, {3, 2},{3, 3}, {3, 4},
    //              {4, -3}, {4, -2},{4, -1},{4, 0}, {4, 1}, {4, 2},{4, 3}, {4, 4},
    // };
    vector<vector<int> >direction(16*fa*fa, vector<int>(2,0));
    int count = 0;
    for(int g=-(2*fa-1);g<=2*fa;g++){
        for(int h=-(2*fa-1);h<=2*fa;h++){
            direction[count][0] = g;
            direction[count][1] = h;
            count++;
        }
    }
    
    for(int i=0;i<16*fa*fa;i++){
        for(int x = 1;x<=sizeRow;x++){
            for (int y = 1; y <= sizeCol; y++)
            {
                int tmpX = x + direction[i][0];
                int tmpY = y + direction[i][1];
                res[i][(x-1)*sizeCol+(y-1)] = cdouble(check(tmpX,tmpY,sizeRow,sizeCol) * 1,0);
                
            }
            
        }
    }

    vector<vector<vector<cdouble>>> tmp(16*fa*fa, vector<vector<cdouble>>(cipherNum, vector<cdouble>(slots, cdouble(0,0))));
    for(int i=0;i<16*fa*fa;i++){
        for(int x = 0; x<cipherNum-1;x++){
            for (int y = 0; y < slots; y++){
                tmp[i][x][y] = res[i][x*slots+y];
            }
        }
        for(int j=(cipherNum-1)*slots; j<sizeRow*sizeCol; j++){
            tmp[i][cipherNum-1][j-(cipherNum-1)*slots] = res[i][j];
        }
    }
    return tmp;
}

// vector<cdouble *> multigenBicleft(int slots){
//     vector<int> rotstep(16*fa*fa);
//     int tmp;
//     for(int g=0; g<2*fa; g++){
//         tmp = (2*fa-1-g) * globalSizeCol + (2*fa-1);
//         for(int h=0; h<BickernelCol; h++){
//             if(tmp-i<0){
//                 rotstep[g*BickernelCol+h] = i-tmp;
//             }
//             rotstep[g*BickernelCol+h] = tmp-i;
//         }
//     }  
//     for(int g=2*fa; g<4*fa; g++){
//         tmp = (g-2*fa+1) * globalSizeCol - (2*fa-1);
//         for(int h=0; h<BickernelCol; h++){
//             rotstep[g*BickernelCol+h] = tmp+i;
//         }
//     }

//     vector<cdouble *> res(16*fa*fa);
//     for (int k=0; k<16*fa*fa;k++){
//         cdouble * V1 = new cdouble[slots];
//         cdouble * V2 = new cdouble[slots];
//         for (int i=slots-rotstep[k]; i < slots; i++){
//             V1[i] = cdouble(0,0);
//             V2[i] = cdouble(1,0);
//         }
//         for(int i=0; i<(slots-rotstep[k]); i++){
//             V1[i] = cdouble(1,0);
//             V2[i] = cdouble(0,0);
//         }
//         res[k] = V1;
//     }
//     return res;
// }


// Initialize parameters
void InitializeParameter(long &logq, long &logp, long &logn,long &logpc,long &slots, int sizeRow,int sizeCol){
    logq = 880; ///< Ciphertext Modulus  //1200
    logp = 30; ///< Real message will be quantized by multiplying 2^40
    // logn = getLeastLargerSecondPower(sizeRow * sizeCol); ///< log2(The number of slots) 
    logn = 14;
    logpc = 30;
    slots = 1 << logn;

    
    cipherNum = ceil(((double)sizeRow*sizeCol)/((double)slots));             

    // cout << "weightmask" << endl;
    globalWeightMasks = genWeightMasks(slots,sizeRow,sizeCol); 
    // cout << "Bicmask" << endl; // Need to modify (only function parameters modified here)
    BicWegihtMasks = genBicWeightMaska(slots,sizeRow,sizeCol);
    multiglobalWeightMasks = multigenWeightMasks(slots, sizeRow,sizeCol,cipherNum);
    multiBicWegihtMasks = multigenBicWeightMaska(slots, sizeRow,sizeCol,cipherNum);
}

template<typename T>
T * vectorMatrixToPtrForslots(vector<vector<T> >&matrix,int slots){
    int n = matrix.size(), m = matrix[0].size();
    T * tmpPtr = new T[slots];
    memset(tmpPtr, 0, slots * sizeof(T));
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            tmpPtr[i*m+j] = matrix[i][j];
        }
        
    }
    return tmpPtr;
}




vector<cdouble> doubleToComplexVector(vector<double>&vec){
    vector<cdouble> res;
    for(double data:vec){
        res.push_back(cdouble(data,0));
    }
    return res;
}

template<typename T>
void displayPtr(T * ptr,int row,int col){
    for(int i=0;i<row*col;i+=col){
        cout << ptr[i] << " ";
    }cout << endl;
}

// /////////////////////////////////////////////////////////////////////////
/////////////   Multi-ciphertext scheme   //////////////////////////////////////////////

vector<vector<vector<double>>> multigetMatrixFromPath(char * path, int sizeRow,int sizeCol){
    fstream file;
    file.open(path,ios::in|ios::out);

    vector<vector<vector<double>>> tmp(inputchannel, vector<vector<double>>(sizeRow, vector<double>(sizeCol,0)));

    for (int k=0; k<inputchannel; k++){
        for(int i = 0; i < sizeRow; i++){
            for(int j = 0; j < sizeCol; j++){
                file >> tmp[k][i][j];
            }
        }
    }

    file.close();
    return tmp;
} 

vector<vector<vector<cdouble>>>  multidoubleToComplex(vector<vector<vector<double>> >&image){
    vector<vector<vector<cdouble>>> tmp (inputchannel,vector<vector<cdouble>>(image[0].size(),vector<cdouble>(image[0][0].size(),0)));

    for (int k=0; k<inputchannel;k++){
        for(int i=0;i<image[0].size();i++){
            for(int j=0; j<image[0][0].size(); j++){
                tmp[k][i][j] = cdouble(image[k][i][j],0);
            }
        }
    }

    return tmp;
}

template<typename T>
vector<vector<T *>> multivectorMatrixToPtrForslots(vector<vector<vector<T>> >&matrix,int slots, int cipherNum){
    int n = matrix[0].size(), m = matrix[0][0].size();
    vector<vector<T>> tmp(inputchannel, vector<T>(m*n));

    for (int k=0; k<inputchannel;k++){
        for(int i=0;i<n;i++){
            for(int j=0; j<m; j++){
                tmp[k][i*m+j] = matrix[k][i][j];
            }
        }
    }
    vector<vector<T *>> res(inputchannel, vector<T *>(cipherNum));
    for (int k=0; k<inputchannel;k++){
        for(int o=0; o<cipherNum-1; o++){
            T * tmpPtr = new T[slots];
            memset(tmpPtr, 0, slots * sizeof(T));
            for(int i=0; i<slots; i++){
                tmpPtr[i] = tmp[k][o*slots+i];
            }
            res[k][o] = tmpPtr;
        }
        T * tmpPtr = new T[slots];
        memset(tmpPtr, 0, slots * sizeof(T));
        for(int i=(cipherNum-1)*slots; i<m*n; i++){
            tmpPtr[i-(cipherNum-1)*slots] = tmp[k][i];
        }
        res[k][cipherNum-1] = tmpPtr;
    }
    return res;
}



/////////////   Bicubic interpolation   //////////////////////////////////////////////
// double compute_W(double x){
//     double res;
//     double a = -0.5;
//     if(-1 <= x <= 1){
//         res = (a+2)*abs(x)*abs(x)*abs(x)-(a+3)*abs(x)*abs(x)+1;
//     }
//     else if(1<abs(x)<2){
//         res = a*abs(x)*abs(x)*abs(x)-5*a*abs(x)*abs(x)-4*a;
//     }
//     else
//     res = 0;
//     return res;
// }

// double abs(double x){
//     if (x <= 0){
//         return -x;
//     }
//     else
//     return x;
// }


// vector<Ciphertext> Bicu(Scheme & scheme, vector<Ciphertext> image, int scale, int sizeRow, int sizeCol, int slots){
//     // vector<Ciphertext> tmpCipher(image);
//     Ciphertext zero;
//     scheme.encryptZeros(zero,slots,logp,logq);
//     vector<Ciphertext> res(inputchannel,zero)
//     int initr, initc;
//     double w_i, w_j;
//     // cdouble * w = new cdouble[slots];             
//     initr = sizeRow / scale;
//     initc = sizeCol / scale;
//     // cout << "1 " << endl;
//     for(int k=0; k<inputchannel; k++){
//         vector<Ciphertext> res(sizeRow*sizeCol/(scale*scale), zero);
//         for(int i = 0; i < sizeRow; i++){
//             for(int j = 0; j < sizeCol; j++){
//                 double x = i * initr/sizeRow;
//                 double y = j * initc/sizeCol;
//                 int x1 = (int)x;
//                 int y1 = (int)y;
//                 Ciphertext sum(zero);
//                 // cout << "2 " << endl;
//                 for(int m = 0; m <= 3; m++){
//                     for(int n = 0; n <= 3; n++){
//                         if (x1+m-1>=0 and y1+n-1>=0 and x1+m-1<srcH and y1+n-1<srcW){
//                             w_i = compute_W(x-m);
//                             w_j = compute_W(y-n);
//                             cdouble * w = new cdouble[slots];
//                             pos = (x1+m-1) * initc +(y1+n-1)
//                             w[pos] = cdouble(1,0) * w_i * w_j;
//                             Ciphertext tmp(image[k]);
//                             scheme.multByConstVecAndEqual(tmp, w, logp);
//                             scheme.reScaleByAndEqual(tmp, logp);
//                             scheme.modDownToAndEqual(sum, tmp.logq);
//                             scheme.addAndEqual(sum, tmp);
//                         }

                        
//                     }
//                 }
//                 scheme.modDownToAndEqual(res[(i-1)*finc+j-1], sum.logq);
//                 scheme.addAndEqual(res[(i-1)*finc+j-1], sum);
//             }
//         }
//     }
//     return res;
// }


// vector<vector<Ciphertext>> multiBicu(Scheme & scheme, vector<vector<Ciphertext>> &image, int scale, int sizeRow, int sizeCol, int slots, int cipherNum){
//     vector<vector<Ciphertext>> tmpCipher(image);
//     vector<Ciphertext> zero(cipherNum);
//     for (int i=0; i<cipherNum; i++){
//         scheme.encryptZeros(zero[i],slots,logp,logq);
//     }
//     vector<vector<Ciphertext>> res(inputchannel, zero);
//     int finr, finc;
//     double w_i, w_j;
//     vector<cdouble *> w(cipherNum)
//     for(int i=0; i<cipherNum; i++){
//         cdouble * tmpPtr = new cdouble[slots];
//         w[i] = tmpPtr;
//     }
//     finr = sizeRow * scale;
//     finc = sizeCol * scale; 
//     for(int k=0; k<inputchannel; k++){
//         for(int i = 1; i <= finr; i++){
//             for(int j = 1; j <= finc; j++){
//                 double x = i * sizeRow/finr;
//                 double y = j * sizeCol/finc;
//                 int x1 = (int)x;
//                 int y1 = (int)y;
//                 vector<Ciphertext> sum(zero); 
//                 for(int m = 0; m <= 3; m++){
//                     for(int n = 0; n <= 3; n++){
//                         w_i = compute_W(x-m);
//                         w_j = compute_W(y-n);
//                         w[0] = cdouble(1,0);
//                         Ciphertext tmp(zero);
//                         scheme.multByConstVecAndEqual(tmp, w, logp);
//                         scheme.reScaleByAndEqual(tmp, logp);
//                         scheme.modDownToAndEqual(sum, tmp.logq);
//                         scheme.addAndEqual(sum, tmp);
//                     }
//                 }
//                 scheme.modDownToAndEqual(res[(i-1)*finc+j-1], sum.logq);
//                 scheme.addAndEqual(res[(i-1)*finc+j-1], sum);
//                 cout << (i-1)*finc+j-1 << endl;
//             }
//         }
//     }
    
//     return res;
// }
/////////////   Bicubic interpolation   //////////////////////////////////////////////

/////////////   Bilinear interpolation   //////////////////////////////////////////////
// vector<Ciphertext> Bili(Scheme & scheme, vector<Ciphertext> image, int scale, int sizeRow, int sizeCol, int slots){
//     vector<Ciphertext> tmpCipher(image);
//     Ciphertext zero;
//     scheme.encryptZeros(zero,slots,logp,logq);
//     vector<Ciphertext> res(sizeRow*scale*sizeCol*scale, zero);
//     int finr, finc;
//     cdouble * w = new cdouble[slots];
//     finr = sizeRow * scale;
//     finc = sizeCol * scale;
//     for(int i = 0; i < finr; i++){
//         for(int j = 0; j < finc; j++){
//             double x = i * sizeRow/finr;
//             double y = j * sizeRow/finc;
//             int x1 = (int)x;
//             int y1 = (int)y;
//             x = x-x1;
//             y = y-y1;
//             if(x1<0){
//                 x = 0;
//                 x1 = 0;
//             }
//             if(x1>=sizeRow-1){
//                 x = 1;
//                 x1 = sizeRow-2;
//             }
//             if(y1<0){
//                 y = 0;
//                 y1 = 0;
//             }
//             if(y1>=sizeCol-1){
//                 y = 1;
//                 y1 = sizeCol-2;
//             }
//             Ciphertext sum(zero);
//             for(int k=0; k < 4; k++){
//                 w[0] = cdouble(1, 0);
//                 Ciphertext tmp(zero);
//                 scheme.multByConstVecAndEqual(tmp, w, logp);
//                 scheme.reScaleByAndEqual(tmp, logp);
//                 scheme.modDownToAndEqual(sum, tmp.logq);
//                 scheme.addAndEqual(sum, tmp);
//             }
//             scheme.modDownToAndEqual(res[i*finc+j], sum.logq);
//             scheme.addAndEqual(res[i*finc+j], sum);
//             cout << i*finc+j << endl;
//         }
//     }
//     return res;
// }
/////////////   Bilinear interpolation   //////////////////////////////////////////////





#endif

/*

\begin{figure}
\centering
\subcaptionbox{高分辨率图片‘蝴蝶’}{
\includegraphics[width=0.2\linewidth]{flyGrayfly.png}
}
\hfill
\subcaptionbox{高分辨率图片‘鸟’}{
\includegraphics[width=0.2\linewidth]{birdGrayBird.png}
}
\hfill
\subcaptionbox{高分辨率图片‘头’}{
\includegraphics[width=0.2\linewidth]{headGrayHead.png}
}
\hfil

\subcaptionbox{32$\times$32的低分辨率图片}{
\includegraphics[width=0.05\linewidth]{flyBicu32.png}
}
\hfill
\subcaptionbox{32$\times$32的低分辨率图片}{
\includegraphics[width=0.05\linewidth]{birdBicu32.png}
}
\hfill
\subcaptionbox{32$\times$32的低分辨率图片}{
\includegraphics[width=0.05\linewidth]{headBicu32.png}
}
\hfill

\subcaptionbox{放大四倍生成的高分辨率图片}{
\includegraphics[width=0.2\linewidth]{fly32Lr2Hr.png}
}
\subcaptionbox{放大四倍生成的高分辨率图片}{
\includegraphics[width=0.2\linewidth]{bird32Lr2Hr.png}
}
\subcaptionbox{放大四倍生成的高分辨率图片}{
\includegraphics[width=0.2\linewidth]{head32Lr2Hr.png}
}

\subcaptionbox{43$\times$43的低分辨率图片}{
\includegraphics[width=0.2\linewidth]{flyBicu43.png}
}
\subcaptionbox{43$\times$43的低分辨率图片}{
\includegraphics[width=0.2\linewidth]{birdBicu43.png}
}
\subcaptionbox{43$\times$43的低分辨率图片}{
\includegraphics[width=0.2\linewidth]{headBicu43.png}
}

\subcaptionbox{放大三倍生成的高分辨率图片}{
\includegraphics[width=0.2\linewidth]{fly43Lr2Hr.png}
}
\subcaptionbox{放大三倍生成的高分辨率图片}{
\includegraphics[width=0.2\linewidth]{bird43Lr2Hr.png}
}
\subcaptionbox{放大三倍生成的高分辨率图片}{
\includegraphics[width=0.2\linewidth]{head43Lr2Hr.png}
}

\subcaptionbox{64$\times$64的低分辨率图片}{
\includegraphics[width=0.2\linewidth]{flyBicu64.png}
}
\subcaptionbox{64$\times$64的低分辨率图片}{
\includegraphics[width=0.2\linewidth]{birdBicu64.png}
}
\subcaptionbox{64$\times$64的低分辨率图片}{
\includegraphics[width=0.2\linewidth]{headBicu64.png}
}

\subcaptionbox{放大两倍生成的高分辨率图片}{
\includegraphics[width=0.2\linewidth]{fly64Lr2Hr.png}
}
\subcaptionbox{放大两倍生成的高分辨率图片}{
\includegraphics[width=0.2\linewidth]{bird64Lr2Hr.png}
}
\subcaptionbox{放大两倍生成的高分辨率图片}{
\includegraphics[width=0.2\linewidth]{head64Lr2Hr.png}
}

\caption{不同数据集下不同方法生成图像的PSNR}
\label{fig:figure}
\end{figure}
*/