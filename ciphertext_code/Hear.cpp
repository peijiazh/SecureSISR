#include <cmath>
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
#include "myUtils.cpp"
#include "settings.cpp"
#include "model.cpp"
#include "testUtils.cpp"

#include <unistd.h>
#include <sys/resource.h>
#include <bits/stdc++.h>
#include <sys/sysinfo.h>
#include <omp.h>

using namespace NTL;
using namespace std;

typedef complex<double> cdouble;

int main(int argc, char * argv []){
    SetNumThreads(get_nprocs());
    // 1. Check parameters
    if(argc < 4) warningAndExit();
    char *LRImaegePath = argv[1];
    int sizeRow = atoi(argv[2]);
    int sizeCol = atoi(argv[3]);

    cout << sizeRow << " --sizeRow And sizeCol-- "<<sizeCol<<"\n";
    globalSizeRow = sizeRow;
    globalSizeCol = sizeCol;

    

    // 2. Initialize encryption parameters
    cout << "init" << endl;
    InitializeParameter(logq, logp, logn,logpc,slots, sizeRow,sizeCol);  

    // 3. First load low-resolution image, path specified by first parameter
    // vector<vector<double> > image = getMatrixFromPath(LRImaegePath, sizeRow, sizeCol);  
    // vector<vector<cdouble> > CImage = doubleToComplex(image);
    // cdouble * CImagePtr = vectorMatrixToPtrForslots(CImage,slots); 
    /// Multi-ciphertext 
    vector<vector<vector<double>>> image = multigetMatrixFromPath(LRImaegePath, sizeRow, sizeCol);          //init
    vector<vector<vector<cdouble>>> CImage = multidoubleToComplex(image);
    vector<vector<cdouble *>> CImagePtr = multivectorMatrixToPtrForslots(CImage, slots, cipherNum);




    // 4. Embed all complex images into ciphertexts
    // 4.1 Initialize encryption parameters
    Ring ring;
	SecretKey secretKey(ring);
	Scheme scheme(secretKey, ring);
    scheme.addLeftRotKeys(secretKey);
    scheme.addRightRotKeys(secretKey);

    // 4.2 Embed CimagePtr into ciphertexts
    ///////////////   Original ////////////////////////////////////////
    // Ciphertext cipherImage;
    // timeutils.start("encrypt...");
    // for(int i=0; i<inputchannel; i++){
    //     Ciphertext cipherImage;
    //     scheme.encrypt(cipherImage,CImagePtr[0][0], slots, logp, logq);
    // }
    
    // // scheme.encrypt(cipherImage,CImagePtr, slots, logp, logq);
    // timeutils.stop("encrypt...");

    timeutils.start("encrypt...");                                                                   
    vector<vector<Ciphertext>> cipherImages(inputchannel, vector<Ciphertext>(cipherNum));
    for (int i=0; i<inputchannel; i++){
        for (int j=0; j<cipherNum; j++){
            Ciphertext cipherImage;
            cdouble * tmpPtr = CImagePtr[i][j];
            scheme.encrypt(cipherImage, tmpPtr, slots, logp, logq);
            scheme.encryptZeros(cipherImages[i][j],slots,logp,logq);
            scheme.modDownToAndEqual(cipherImages[i][j],cipherImage.logq);
            scheme.addAndEqual(cipherImages[i][j],cipherImage);
        }
    }
    timeutils.stop("encrypt...");

    ///////////////   Original ////////////////////////////////////////

    

    

    //5. Computation
    cipherNet cn(scheme,activationParams,sizeRow,sizeCol,secretKey);
    // vector<Ciphertext>cipherImages(1,cipherImage);
    timeutils.start("network computing...");

    ///// Multi-ciphertext ////////////    
    // First work                                                                                     
    // cout << "Bic" << endl;
    // timeutils.start("Bicconv...");
    // vector<vector<Ciphertext>> cTmps = cn.multiBicConv(cipherImages, BicKerweight,inputchannel, cipherNum);
    // timeutils.stop("Bicconv...");   
    // // timeutils.start("Bicconv...");
    // // vector<vector<Ciphertext>> cTmps = cn.Bicu(scheme, cipherImages, fa, sizeRow, sizeCol, slots);
    // // timeutils.stop("Bicconv...");

    // vector<vector<Ciphertext>> residual(cTmps);
    // for(int d=0; d<8;d++){
    //     // vector<vector<Ciphertext>> cTmps(cipherImages);
    //     // vector<vector<Ciphertext>> cTmps(64, vector<Ciphertext>(cipherNum));
    //     // for (int i = 0; i < cTmps.size(); i++){
    //     //     for (int k=0; k<cipherNum; k++){
    //     //         scheme.encryptZeros(cTmps[i][k],slots,logp,logq);
    //     //     }
    //     // }
    //     cout << "conv " << d <<endl;
    //     timeutils.start("Conv...");
    //     cTmps = cn.multiconv(cTmps,d,cipherNum,slots);
    //     timeutils.stop("Conv...");
  
    //     // cout << "relu " << endl;
    //     timeutils.start("Relu...");
    //     cn.multirelu(cTmps,cipherNum);
    //     timeutils.stop("Relu...");
    // }

    // // cout << "conv8 " << endl;
    // timeutils.start("Conv8...");
    // cTmps = cn.multiconv(cTmps,8,cipherNum,slots);
    // timeutils.stop("Conv...");

    // // cout << "add " << endl;
    // timeutils.start("ADD...");
    // for(int d=0; d < cTmps.size(); d++){
    //     for(int e=0;e<cTmps[0].size();e++){
    //         scheme.modDownToAndEqual(residual[d][e],cTmps[d][e].logq);
    //         scheme.addAndEqual(cTmps[d][e],residual[d][e]);
    //     }
    // }
    // cout << getCurrentRSS()<<"GB" <<endl;
    // timeutils.stop("ADD...");

    //////////////////// New structure + pixel (second work) //////////////////////////////////////////////////////////
    // cout << "input " <<endl;
    // timeutils.start("input...");
    // vector<vector<Ciphertext>> cTmps = cn.multiconv(cipherImages,0,cipherNum,slots);
    // timeutils.stop("input...");

    // vector<vector<Ciphertext>> inputs(cTmps);
    // for(int d=1; d<=3;d++){
    //     cout << "conv " << d <<endl;
    //     timeutils.start("Conv3...");
    //     cTmps = cn.multiconv(cTmps,d,cipherNum,slots);
    //     timeutils.stop("Conv3...");
  
    //     timeutils.start("Relu...");
    //     cn.multirelu(cTmps,cipherNum);
    //     timeutils.stop("Relu...");
    // }

    // vector<vector<Ciphertext>> inputs1(inputs);
    // timeutils.start("ADD...");
    // for(int d=0; d < cTmps.size(); d++){
    //     for(int e=0;e<cTmps[0].size();e++){
    //         scheme.modDownToAndEqual(inputs1[d][e],cTmps[d][e].logq);
    //         scheme.addAndEqual(cTmps[d][e],inputs1[d][e]);
    //     }
    // }
    // timeutils.stop("ADD...");

    // timeutils.start("conv1...");
    // cTmps = cn.multiconv1x1(cTmps,4,cipherNum);
    // timeutils.stop("conv1...");

    // vector<vector<Ciphertext>> midout(cTmps);
    // for(int d=5; d<=7;d++){
    //     cout << "conv " << d <<endl;
    //     timeutils.start("Conv1...");
    //     cTmps = cn.multiconv1x1(cTmps,d,cipherNum);
    //     timeutils.stop("Conv1...");
  
    //     timeutils.start("Relu...");
    //     cn.multirelu(cTmps,cipherNum);
    //     timeutils.stop("Relu...");
    // }

    // cout << "mul" << endl;
    // timeutils.start("Mul...");
    // for(int d=0; d < cTmps.size(); d++){
    //     for(int e=0;e<cTmps[0].size();e++){
    //         scheme.modDownToAndEqual(midout[d][e],cTmps[d][e].logq);
    //         scheme.multAndEqual(cTmps[d][e], midout[d][e]);
    //         scheme.reScaleByAndEqual(cTmps[d][e],logp);
    //     }
    // }
    // timeutils.stop("Mul...");

    // cout << "conv " << 8 <<endl;
    // timeutils.start("Conv3...");
    // cTmps = cn.multiconv(cTmps,8,cipherNum,slots);
    // timeutils.stop("Conv3...");

    // timeutils.start("ADD...");
    // for(int d=0; d < cTmps.size(); d++){
    //     for(int e=0;e<cTmps[0].size();e++){
    //         scheme.modDownToAndEqual(inputs[d][e],cTmps[d][e].logq);
    //         scheme.addAndEqual(cTmps[d][e],inputs[d][e]);
    //     }
    // }
    // timeutils.stop("ADD...");

    // timeutils.start("output...");
    // cTmps = cn.multiconv(cTmps,9,cipherNum,slots);
    // timeutils.stop("output...");
    // timeutils.start("pixel...");
    // cTmps = cn.multipixel(cTmps,fa,cipherNum,slots);
    // timeutils.stop("pixel...");

    //////////////////// Original structure + pixel (second work) //////////////////////////////////////////////////////////
    cout << "input " <<endl;
    timeutils.start("input...");
    vector<vector<Ciphertext>> cTmps = cn.multiconv(cipherImages,0,cipherNum,slots);
    timeutils.stop("input...");
    timeutils.start("Relu...");
    cn.multirelu(cTmps,cipherNum);
    timeutils.stop("Relu...");

    vector<vector<Ciphertext>> inputs(cTmps);
    for(int d=1; d<=7;d++){
        cout << "conv " << d <<endl;
        timeutils.start("Conv3...");
        cTmps = cn.multiconv(cTmps,d,cipherNum,slots);
        timeutils.stop("Conv3...");
  
        timeutils.start("Relu...");
        cn.multirelu(cTmps,cipherNum);
        timeutils.stop("Relu...");
    }

    timeutils.start("ADD...");
    for(int d=0; d < cTmps.size(); d++){
        for(int e=0;e<cTmps[0].size();e++){
            scheme.modDownToAndEqual(inputs[d][e],cTmps[d][e].logq);
            scheme.addAndEqual(cTmps[d][e],inputs[d][e]);
        }
    }
    timeutils.stop("ADD...");

    timeutils.start("output...");
    cTmps = cn.multiconv(cTmps,8,cipherNum,slots);
    timeutils.stop("output...");
    timeutils.start("pixel...");
    cTmps = cn.multipixel(cTmps,fa,cipherNum,slots);
    timeutils.stop("pixel...");

     ///// Multi-ciphertext ////////////
    
    // cout << "conv1 " << endl;
    // cTmps = cn.multiconv(cTmps,8,cipherNum,slots);

    // vector<Ciphertext> cipherImages = {cipherImage, cipherImage, cipherImage, cipherImage};
    // vector<Ciphertext> cTmps = cn.forward(cipherImages);
    // // Ciphertext cTmp = cn.forward(cipherImages);
    // vector<Ciphertext> resudial(cipherImages);
    // cout << "conv0 " << endl;
    // vector<Ciphertext> cTmps = cn.conv(cipherImages,0);
    // cout << "relu0 " << endl;
    // cn.relu(cTmps);
    // cout << "conv1 " << endl;
    // cTmps = cn.conv(cTmps,1); 
    // cout << "relu1 " << endl;
    // cn.relu(cTmps);
    // cout << "conv2 " << endl;
    // cTmps = cn.conv(cTmps,2); 
    // cout << "relu2 " << endl;
    // cn.relu(cTmps);
    // cout << "add " << endl;
    // for(int k=0; k < cTmps.size(); k++)
    // {
    //     scheme.modDownToAndEqual(resudial[0],cTmps[k].logq);
    //     scheme.addAndEqual(cTmps[k],resudial[0]);
    //     resudial = cipherImages;
    // }
    // cout << "conv3 " << endl;
    // cTmps = cn.conv(cTmps,3); 
    // cout << "relu3 " << endl;
    // cn.relu(cTmps);
    // cout << "conv4 " << endl;
    // cTmps = cn.conv(cTmps,4); 
    // cout << "relu4 " << endl;
    // cn.relu(cTmps);
    // cout << "conv5 " << endl;
    // cTmps = cn.conv(cTmps,5); 
    // cout << "relu5 " << endl;
    // cn.relu(cTmps);
    // for(int k=0; k < cTmps.size(); k++)
    // {
    //     scheme.modDownToAndEqual(resudial[0],cTmps[k].logq);
    //     scheme.addAndEqual(cTmps[k],resudial[0]);
    //     resudial = cipherImages;
    // }
    // cout << "conv6 " << endl;
    // cTmps = cn.conv(cTmps,6); 
    // cout << "relu6 " << endl;
    // cn.relu(cTmps);
    // cout << "conv7 " << endl;
    // cTmps = cn.conv(cTmps,7); 
    // cout << "relu7 " << endl;
    // cn.relu(cTmps);
    // cout << "conv8 " << endl;
    // cTmps = cn.conv(cTmps,8);
    // cout << "add " << endl;
    // for(int k=0; k < cTmps.size(); k++)
    // {
    //     scheme.modDownToAndEqual(resudial[0],cTmps[k].logq);
    //     scheme.addAndEqual(cTmps[k],resudial[0]);
    //     resudial = cipherImages;
    // }
    // cout << "pixel " << endl;
    // Ciphertext cTmp = cn.pixel(cTmps,2);
    // vector<Ciphertext>cTmp1(1,cTmp);
    timeutils.stop("network computing...");
    

    //6. Decryption
    ///////////////   Original ////////////////////////////////////////
    // int level = 8;
    // fstream file;
    // // const char * savePath = "finalOuputFortimeComsuming.txt";
    // const char * savePath = "my_finaloutput.txt";
    // file.open(savePath,ios::in|ios::out);
    // timeutils.start("decrypt...");
    
    // cdouble * res = scheme.decrypt(secretKey,cTmp1[0]);
    // timeutils.stop("decrypt...");
    // for (int i = 0; i < sizeRow; i++)
    // {
        // for(int j=0;j< sizeCol;j++){
            // cdouble * finres = scheme.decrypt(secretKey,cipherImages[0]);
            // cout << res[i*sizeRow+j].real() <<" "; 
        // }cout << "\n";
    // }
    // timeutils.stop("decrypt...");
    // cout << "finish " << endl;
    // file.close();

//////////////// Multi-ciphertext output ///////////////////////////////////////////                                          
    int level = 8;
    fstream file;
    const char * savePath = "my_finaloutput.txt";
    file.open(savePath,ios::in|ios::out);
    timeutils.start("decrypt...");
    vector<vector<cdouble>> tmp(inputchannel, vector<cdouble>(slots*cipherNum, 0));
    for (int k=0; k<inputchannel; k++){
        for(int o=0; o <cipherNum; o++){
            cdouble * res = scheme.decrypt(secretKey,cTmps[k][o]);
            for(int i=0; i <slots; i++){
                tmp[k][o*slots+i] = res[i];
            }
        }
    }
    
    for (int k=0; k<inputchannel; k++){
        for (int i = 0; i < sizeRow*fa; i++){
            for(int j=0;j< sizeCol*fa;j++){
                file << tmp[k][i*sizeCol*fa+j].real() <<" ";
            }file << "\n";
        }
    }
    // cdouble * res = scheme.decrypt(secretKey,cipherImage);
    // for (int i = 0; i < sizeRow; i++){
    //     for(int j=0;j< sizeCol;j++){
    //         file << res[i*sizeCol+j].real() <<" ";
    //     }file << "\n";
    // }
    timeutils.stop("decrypt...");
    cout << "finish " << endl;
    file.close();
    //////////////// Multi-ciphertext output ///////////////////////////////////////////
    ///////////////   Original ////////////////////////////////////////



    // 7 Debugging
    // 7.1 Debug convOnceByKernel3 interface
    // testConvOnceOneImageOneKernel(scheme,secretKey,LRImaegePath);
    // 7.2 Debug convOnce interface
    // testConvOnceOneImage(scheme,secretKey,LRImaegePath);
    // 7.3 Test if one convolution + activation is normal
    // testForward(scheme,secretKey,LRImaegePath);
    // 7.3 Test if one pixel is normal
    // testpixel(scheme, secretKey, LRImaegePath);

    return 0;
}