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

#include <unistd.h>
#include <sys/resource.h>
#include <bits/stdc++.h>
#include <omp.h>

using namespace NTL;
using namespace std;


#ifndef __MODEL_CPP__
#define __MODEL_CPP__

class cipherNet{
public:
    vector<Ciphertext> forward(vector<Ciphertext> ct){
        int turn = 8;
        cout << "row col " << globalSizeRow << " "<< globalSizeCol <<endl;
        vector<Ciphertext> tmpCipher(ct);
        vector<Ciphertext> resudial(ct);
        // 1. Perform 8 convolutions + activation
        for (int i = 0; i < turn; i++)
        {
            // 1.1 Convolution
            timeutils.start("conv...");
            cout << "index = "<< i <<endl;
            tmpCipher = conv(tmpCipher,i); // Function modified here
            timeutils.stop("One Conv times...");

            // 1.2 ReLU activation
            timeutils.start("relu...");
            relu(tmpCipher);
            timeutils.stop("One relu times...");
            // cout << getCurrentRSS()<<"GB" <<endl;
            // if(i == 0) exit(0);
            if(i == 2 || i == 5)
            {
                timeutils.start("add...");
                for(int k=0; k < tmpCipher.size(); k++)
                {
                    scheme.modDownToAndEqual(resudial[0],tmpCipher[k].logq);
                    scheme.addAndEqual(tmpCipher[k],resudial[0]);
                    resudial = ct;
                }
                timeutils.stop("One add times...");
            }

        }

        // 2. Final convolution
        timeutils.start("conv...");
        cout << "index = "<< 8 <<endl;
        tmpCipher = conv(tmpCipher,8);
        timeutils.stop("One Conv times...");
        cout << tmpCipher.size() << " after 9 conv"<<endl; 

        // 3. Addition
        timeutils.start("add...");
        for (int i = 0; i < tmpCipher.size(); i++)
        {
            scheme.modDownToAndEqual(resudial[0],tmpCipher[i].logq);
            scheme.addAndEqual(tmpCipher[i],resudial[0]);
            resudial = ct;
            // scheme.modDownToAndEqual(ct[0],tmpCipher[i].logq);
            // scheme.addAndEqual(tmpCipher[i],ct[0]);
        }
        timeutils.stop("add times...");

        timeutils.start("pixel computing...");
        Ciphertext cTmp = pixel(tmpCipher,2);
        timeutils.stop("pixel computing...");
        vector<Ciphertext>res(1,cTmp);
             
        return res;
    }
public:
    vector<Ciphertext> conv(vector<Ciphertext>image,int level){
        vector<vector<double> >weights(outChanels[level],vector<double>(inChanels[level] * kernelSize * kernelSize,0));
        loadParams(weights,level);

        vector<Ciphertext> res(outChanels[level]);
        for (int i = 0; i < res.size(); i++)
        {
            scheme.encryptZeros(res[i],slots,logp,logq);
        }

        int gapLen = kernelSize * kernelSize;

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < outChanels[level]; i++)
        {
            
            // Initialize to zero
            Ciphertext cipher;
            scheme.encryptZeros(cipher, slots, logp, logq);

            int j;
            for ( j = 0; j < inChanels[level]; j++)
            {
                // One image per in_channel
                Ciphertext tmp = convOnceByKernel3(image[j],weights[i],j*gapLen,(j+1)*gapLen-1);// Function modified here

                scheme.modDownToAndEqual(cipher,tmp.logq);
                scheme.addAndEqual(cipher,tmp);
            }
            
            scheme.modDownToAndEqual(res[i],cipher.logq);
            scheme.addAndEqual(res[i],cipher);

        }

        cout << getCurrentRSS()<<"GB" <<endl;
        
        return res;
    }

    // Convolution with 3*3 kernel
    Ciphertext convOnceByKernel3(Ciphertext &image, vector<double>&doubleWeight,int start,int end){
        Ciphertext res;
    	scheme.encryptZeros(res, slots, logp, logq);

        convOnceByKernel3RowOne(scheme,image,res,doubleWeight,start,start+2,globalSizeCol/2,0);// Only function parameters modified here
        // cout << "RowOne OK..\n";
        convOnceByKernel3RowTwo(scheme,image,res,doubleWeight,start+3,start+5,globalSizeCol/2,3);// Only function parameters modified here
        // cout << "RowTwo OK..\n";
        convOnceByKernel3RowThree(scheme,image,res,doubleWeight,start+6,start+8,globalSizeCol/2,6);// Only function parameters modified here
        // cout << "RowThree OK..\n";

        
        return res;
    }

    // First row of 3*3 convolution
    void convOnceByKernel3RowOne(Scheme & scheme, Ciphertext &image, Ciphertext &res, vector<double>&doubleWeight,int start,int end,int cols,int base){
        Ciphertext tmpRotate(image);
        // Need to move col+1 steps, but not necessarily a power of 2, need to decompose
        vector<int> moveSteps = getMoveSteps(cols+1);
        for (int i = 0; i < moveSteps.size(); i++)
        {
            scheme.rightRotateFastAndEqual(tmpRotate,moveSteps[i]);
        }
        

        for (int i = start; i <= end; i++)
        {
            // Get the corresponding plaintext weight packing
            cdouble * complexWeight = getConvWeightArrange(doubleWeight[i],i-start+base,slots);
            
            Ciphertext tmp(tmpRotate);
            scheme.multByConstVecAndEqual(tmp,complexWeight,logp);
            scheme.reScaleByAndEqual(tmp,logp);
            scheme.modDownToAndEqual(res,tmp.logq);
            scheme.addAndEqual(res,tmp);

            delete []complexWeight;
            // cout << "delete OK\n";
            // Left shift
            scheme.leftRotateFastAndEqual(tmpRotate,1);
            // cout << "leftRotateFastAndEqual OK\n";
        }
    }

    // Second row of 3*3 convolution
    void convOnceByKernel3RowTwo(Scheme & scheme, Ciphertext &image, Ciphertext &res, vector<double>&doubleWeight,int start,int end,int cols,int base){
        Ciphertext tmpRotate(image);

        scheme.rightRotateFastAndEqual(tmpRotate,1);
        
        
        for (int i = start; i <= end; i++)
        {
            // Get the corresponding plaintext weight packing
            cdouble * complexWeight = getConvWeightArrange(doubleWeight[i],i-start+base,slots);

            Ciphertext tmp(tmpRotate);
            scheme.multByConstVecAndEqual(tmp,complexWeight,logp);
            scheme.reScaleByAndEqual(tmp,logp);
            scheme.modDownToAndEqual(res,tmp.logq);
            scheme.addAndEqual(res,tmp);

            delete []complexWeight;
            // Left shift
            scheme.leftRotateFastAndEqual(tmpRotate,1);
        }
    }

    // Third row of 3*3 convolution
    void convOnceByKernel3RowThree(Scheme & scheme, Ciphertext &image, Ciphertext &res, vector<double>&doubleWeight,int start,int end,int cols,int base){
        Ciphertext tmpRotate(image);
        
        // Need to move col-1 steps, but not necessarily a power of 2, need to decompose
        vector<int> moveSteps = getMoveSteps(cols-1);
        for (int i = 0; i < moveSteps.size(); i++)
        {
            scheme.leftRotateFastAndEqual(tmpRotate,moveSteps[i]);
        }
        
        
        for (int i = start; i <= end; i++)
        {
            // Get the corresponding plaintext weight packing
            cdouble * complexWeight = getConvWeightArrange(doubleWeight[i],i-start+base,slots);
            Ciphertext tmp(tmpRotate);
            scheme.multByConstVecAndEqual(tmp,complexWeight,logp);
            scheme.reScaleByAndEqual(tmp,logp);
            scheme.modDownToAndEqual(res,tmp.logq);
            scheme.addAndEqual(res,tmp);

            delete []complexWeight;
            // Left shift
            scheme.leftRotateFastAndEqual(tmpRotate,1);
        }
    }

    // Generate a list representing x as a sum of powers of 2
    vector<int> getMoveSteps(int x){
        int index = 1;
        int tmpCount=0;
        vector<int>res;
        for(int i=0;i<32;i++)
        {
            if(tmpCount >= x) break;

            if((x & index)){
                res.push_back(index);
                tmpCount += (x & index);
            }

            index = index << 1;
        }
        

        return res;
    }

    // Use mask*weight method or pre-arranged vector
    cdouble* getConvWeightArrange(double weight,int index,int slots){
        cdouble * res = new cdouble[slots];
        for (int i = 0; i < slots; i++)
        {
            res[i] = globalWeightMasks[index][i] * weight;
        }
        return res;
    }

    void loadParams(vector<vector<double> >&weights,int level){
        char * path = (char *)weigtsFiles[level].c_str();
        // weights = getMatrixFromPath(path,outChanels[level],inChanels[level]*kernelSize*kernelSize);
        weights = getMatrixFromPath(path,outChanels[level],inChanels[level]*rowKernelsSize[level]*colKernelsSize[level]);

    }

    // Activation function
    void relu(vector<Ciphertext> &image){
        // 6.83593757e-06*(tri(x))+2.34375001e-01*(square(x))+4.99988281e-01*x+1.87499999e-01
        
        // Process each ciphertext in image
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < image.size(); i++)
        {   
            Ciphertext secondLevelCipher,thirdLevelCipher;
            scheme.mult(secondLevelCipher,image[i],image[i]);
            scheme.reScaleByAndEqual(secondLevelCipher,logp);

            // Get cubic ciphertext (optimized to reduce multiplication depth)
            scheme.multByConst(thirdLevelCipher,image[i],actParams[3],logp);
            scheme.reScaleByAndEqual(thirdLevelCipher,logp);
            scheme.multAndEqual(thirdLevelCipher,secondLevelCipher);
            scheme.reScaleByAndEqual(thirdLevelCipher,logp);

            // Get quadratic ciphertext
            scheme.multByConstAndEqual(secondLevelCipher,actParams[2],logp);
            scheme.reScaleByAndEqual(secondLevelCipher,logp);

            // Get linear ciphertext
            scheme.multByConst(image[i],image[i],actParams[1],logp);
            scheme.reScaleByAndEqual(image[i],logp);

            // Add constant term to linear term, add quadratic and cubic terms
            scheme.addConstAndEqual(image[i],actParams[0],logp);
            scheme.addAndEqual(secondLevelCipher,thirdLevelCipher);

            // Adjust logq of firstLevelCipher
            scheme.modDownToAndEqual(image[i], secondLevelCipher.logq );

            // Add both together
            scheme.add(image[i],image[i],secondLevelCipher);
            
        }
    }

    // Sub-pixel convolution
    Ciphertext pixel(vector<Ciphertext> image, int scale){
        Ciphertext res;
    	scheme.encryptZeros(res, slots, logp, logq);
        int pos, channel;
        vector<int> moveSteps;
        for(int i = 0; i < sizeRow; i++){
            for(int j = 0; j < sizeCol; j++){
                pos = (sizeCol/scale) * (i/scale) + (j/scale);
                channel = (i % scale) * scale + (j % scale);
                moveSteps = my_getMoveSteps(pos);
                Ciphertext tmp = computepixel(scheme, image[channel], moveSteps, pos);
                scheme.modDownToAndEqual(res, tmp.logq);
                scheme.addAndEqual(res, tmp);
                scheme.leftRotateFastAndEqual(res, 1); 
            }
        }
        moveSteps = my_getMoveSteps(sizeRow*sizeCol);
        for (int i = 0; i < moveSteps.size(); i++){
                scheme.rightRotateFastAndEqual(res, moveSteps[i]);
            }
        return res;
    }
    
    // Extract pixels from LR in sub-pixel convolution
    Ciphertext computepixel(Scheme & scheme, Ciphertext image, vector<int> moveSteps, int pos){
        Ciphertext res;
        scheme.encryptZeros(res, slots, logp, logq);
        cdouble * mask = new cdouble[slots];
        mask[pos] = cdouble(1,0);
        Ciphertext tmpRotate(image);
        scheme.multByConstVecAndEqual(tmpRotate, mask, logp);
        scheme.reScaleByAndEqual(tmpRotate, logp);
        if(pos != 0){
            for (int i = 0; i < moveSteps.size(); i++){
                scheme.leftRotateFastAndEqual(tmpRotate,moveSteps[i]);
            }
        }
        scheme.modDownToAndEqual(res, tmpRotate.logq);
        scheme.addAndEqual(res, tmpRotate);
        delete []mask;
        return res;
    }


    vector<int> my_getMoveSteps(int x){
        int index = 1;
        int tmpCount=0;
        vector<int>res;
        if(x > 256){
            for(int i = 0; i < (x/256); i++) 
                res.push_back(256);
            
            x = x - (x/256) * 256;
        }
        if(x != 0){
            for(int i=0;i<32;i++){
                if(tmpCount >= x) break;

                if((x & index)){
                    res.push_back(index);
                    tmpCount += (x & index);
                }

                index = index << 1;
            }
        }

        return res;
    }

    

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////// Multi-ciphertext scheme //////////////////////////////////////////////////////////////////////////////////////////
    // Fast upsampling layer
    vector<vector<Ciphertext>> multiBicConv(vector<vector<Ciphertext>> &image, vector<double> doubleWeight, int inputchannel, int cipherNum){
        vector<vector<Ciphertext>> res(inputchannel, vector<Ciphertext>(cipherNum));
        
        for(int k=0; k<inputchannel; k++){
            for(int o=0; o<cipherNum; o++){
                scheme.encryptZeros(res[k][o], slots, logp, logq);
            }
        }

        int rotstep;
        #pragma omp parallel for schedule(static)
        for(int k=0; k<inputchannel; k++){
            for(int i=0; i<2*fa; i++){
                rotstep = (2*fa-1-i) * globalSizeCol + (2*fa-1);
                multiRightBicConv(scheme,image[k],res[k],doubleWeight,i*BickernelCol,rotstep, cipherNum);
            }

            for(int i=2*fa; i<4*fa; i++){
                rotstep = (i-2*fa+1) * globalSizeCol - (2*fa-1);
                multileftBicConv(scheme,image[k],res[k],doubleWeight,i*BickernelCol,rotstep, cipherNum);
            }

        }
        cout << getCurrentRSS()<<"GB" <<endl;
        return res;
    }

    // Computation for first few rows in fast upsampling layer
    void multiRightBicConv(Scheme & scheme, vector<Ciphertext> &image, vector<Ciphertext> &res, vector<double> doubleWeight,int start,int rotstep, int cipherNum){
        
        
        for (int i = 0; i < BickernelCol; i++)
        {
            vector<Ciphertext> tmpRotate(image);
            if((rotstep-i) > 0){
                if(cipherNum == 1){
                    vector<int> moveSteps = getMoveSteps( rotstep-i);
                    for (int o = 0; o < moveSteps.size(); o++){
                        scheme.rightRotateFastAndEqual(tmpRotate[0],moveSteps[o]);
                    }
                }
                else
                    multiRightRot(scheme,tmpRotate,cipherNum, slots, rotstep-i);
            }
            if((rotstep-i) < 0){
                if(cipherNum == 1){
                    vector<int> moveSteps = getMoveSteps(i-rotstep);
                    for (int o = 0; o < moveSteps.size(); o++){
                        scheme.leftRotateFastAndEqual(tmpRotate[0],moveSteps[o]);
                    }
                }
                else
                    multiLeftRot(scheme,tmpRotate, cipherNum, slots, -(rotstep-i));
            }
            vector<cdouble *> complexWeight = multigetBicConvWeightArrange(doubleWeight[i+start],i+start,slots, cipherNum);
            for (int j=0; j<cipherNum; j++){
                Ciphertext tmp(tmpRotate[j]);
                scheme.multByConstVecAndEqual(tmp,complexWeight[j],logp);
                scheme.reScaleByAndEqual(tmp,logp);
                scheme.modDownToAndEqual(res[j],tmp.logq);
                scheme.addAndEqual(res[j],tmp);
                delete []complexWeight[j];
            }
        }
    }

    // Computation for last few rows in fast upsampling layer
    void  multileftBicConv(Scheme & scheme, vector<Ciphertext> &image, vector<Ciphertext> &res, vector<double> doubleWeight,int start,int rotstep, int cipherNum){
        
        
        for (int i = 0; i < BickernelCol; i++)
        {
            vector<Ciphertext> tmpRotate(image);
            if(cipherNum == 1){
                vector<int> moveSteps = getMoveSteps(rotstep+i);
                for (int o = 0; o < moveSteps.size(); o++){
                    scheme.leftRotateFastAndEqual(tmpRotate[0],moveSteps[o]);
                }
            }
            else
                multiLeftRot(scheme,tmpRotate, cipherNum, slots, rotstep+i);
            vector<cdouble *> complexWeight = multigetBicConvWeightArrange(doubleWeight[i+start],i+start,slots, cipherNum);
            for (int j=0; j<cipherNum; j++){
                Ciphertext tmp(tmpRotate[j]);
                scheme.multByConstVecAndEqual(tmp,complexWeight[j],logp);
                scheme.reScaleByAndEqual(tmp,logp);
                scheme.modDownToAndEqual(res[j],tmp.logq);
                scheme.addAndEqual(res[j],tmp);
                delete []complexWeight[j];
            }

            // delete complexWeight;
        }
    }

    // Get weight vector in fast upsampling layer
    vector<cdouble*>  multigetBicConvWeightArrange(double weight,int index,int slots,int cipherNum){
        vector<cdouble *> res(cipherNum);

        for (int k=0; k<cipherNum; k++){
            cdouble * tmp = new cdouble[slots];
            for (int i = 0; i < slots; i++){
                tmp[i] = multiBicWegihtMasks[index][k][i] * weight;
            }
            res[k] = tmp;
        }
        
        return res;
    }

    // Multi-ciphertext right rotation
    void multiRightRot(Scheme & scheme, vector<Ciphertext> &image, int cipherNum, int slots, int rotstep){
        vector<Ciphertext> tmp(image);
        vector<int> moveSteps = getMoveSteps(rotstep);

        for (int o=0; o<cipherNum; o++){
            for (int i = 0; i < moveSteps.size(); i++){
                scheme.rightRotateFastAndEqual(tmp[o],moveSteps[i]);
                scheme.rightRotateFastAndEqual(image[o],moveSteps[i]);
            }
        }

        cdouble * V1 = new cdouble[slots];
        cdouble * V2 = new cdouble[slots];
        for (int i=0; i < rotstep; i++){
            V1[i] = cdouble(0,0);
            V2[i] = cdouble(1,0);
        }
        for (int i=rotstep; i<slots; i++){
            V1[i] = cdouble(1,0);
            V2[i] = cdouble(0,0);
        }

        for(int i=1; i<cipherNum; i++){
            scheme.multByConstVecAndEqual(tmp[i-1],V2,logp);
            scheme.reScaleByAndEqual(tmp[i-1],logp);

            scheme.multByConstVecAndEqual(image[i],V1,logp);
            scheme.reScaleByAndEqual(image[i],logp);

            scheme.modDownToAndEqual(image[i],tmp[i-1].logq);
            scheme.addAndEqual(image[i],tmp[i-1]);
        }

        scheme.multByConstVecAndEqual(tmp[cipherNum-1],V2,logp);
        scheme.reScaleByAndEqual(tmp[cipherNum-1],logp);
        scheme.multByConstVecAndEqual(image[0],V1,logp);
        scheme.reScaleByAndEqual(image[0],logp);
        scheme.modDownToAndEqual(image[0],tmp[cipherNum-1].logq);
        scheme.addAndEqual(image[0],tmp[cipherNum-1]);
        delete []V1;
        delete []V2;

    }

    // Multi-ciphertext left rotation
    void multiLeftRot(Scheme & scheme, vector<Ciphertext> &image, int cipherNum, int slots, int rotstep){
        vector<Ciphertext> tmp(image);
        vector<int> moveSteps = getMoveSteps(rotstep);
        
        for (int o=0; o<cipherNum; o++){
            for (int i = 0; i < moveSteps.size(); i++){
                scheme.leftRotateFastAndEqual(tmp[o],moveSteps[i]);
                scheme.leftRotateFastAndEqual(image[o],moveSteps[i]);
            }
        }

        cdouble * V1 = new cdouble[slots];
        cdouble * V2 = new cdouble[slots];
        for (int i=slots-rotstep; i < slots; i++){
            V1[i] = cdouble(0,0);
            V2[i] = cdouble(1,0);
        }
        for(int i=0; i<(slots-rotstep); i++){
            V1[i] = cdouble(1,0);
            V2[i] = cdouble(0,0);
        }

        for(int i=0; i<cipherNum-1; i++){
            scheme.multByConstVecAndEqual(tmp[i+1],V2,logp);
            scheme.reScaleByAndEqual(tmp[i+1],logp);

            scheme.multByConstVecAndEqual(image[i],V1,logp);
            scheme.reScaleByAndEqual(image[i],logp);

            scheme.modDownToAndEqual(image[i],tmp[i+1].logq);
            scheme.addAndEqual(image[i],tmp[i+1]);
        }

        scheme.multByConstVecAndEqual(tmp[0],V2,logp);
        scheme.reScaleByAndEqual(tmp[0],logp);
        scheme.multByConstVecAndEqual(image[cipherNum-1],V1,logp);
        scheme.reScaleByAndEqual(image[cipherNum-1],logp);
        scheme.modDownToAndEqual(image[cipherNum-1],tmp[0].logq);
        scheme.addAndEqual(image[cipherNum-1],tmp[0]);
        delete []V1;
        delete []V2;
    }

    // Multi-ciphertext convolution
    vector<vector<Ciphertext>> multiconv(vector<vector<Ciphertext>> image, int level, int cipherNum, int slots){
        vector<vector<double> >weights(outChanels[level],vector<double>(inChanels[level] * kernelSize * kernelSize,0));
        loadParams(weights,level);

        vector<vector<Ciphertext>> res(outChanels[level], vector<Ciphertext>(cipherNum));
        for (int i = 0; i < res.size(); i++){
            for (int k=0; k<cipherNum; k++){
                scheme.encryptZeros(res[i][k],slots,logp,logq);
            }
        }

        int gapLen = kernelSize * kernelSize;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < outChanels[level]; i++)
        {
            vector<Ciphertext> cipher(cipherNum);
            for (int k=0; k<cipherNum; k++){
                scheme.encryptZeros(cipher[k],slots,logp,logq);
            }
            
            for (int j = 0; j < inChanels[level]; j++)
            {
                
                vector<Ciphertext> tmp = multiconvOnceByKernel3(image[j],weights[i],j*gapLen,(j+1)*gapLen-1);// Function modified here
                for (int k=0; k<cipherNum; k++){
                    scheme.modDownToAndEqual(cipher[k],tmp[k].logq);
                    scheme.addAndEqual(cipher[k],tmp[k]);
                }
            }

            for (int k=0; k<cipherNum; k++){
                scheme.modDownToAndEqual(res[i][k],cipher[k].logq);
                scheme.addAndEqual(res[i][k],cipher[k]);
            }
        }
        cout << getCurrentRSS()<<"GB" <<endl;
        return res;
    }

    vector<Ciphertext> multiconvOnceByKernel3(vector<Ciphertext> &image, vector<double>&doubleWeight,int start,int end){
        vector<Ciphertext> res(cipherNum);
        for (int k=0; k<cipherNum; k++){
            scheme.encryptZeros(res[k],slots,logp,logq);
        }
        multiconvOnceByKernel3RowOne(scheme,image,res,doubleWeight,start,start+2,globalSizeCol,0);
        multiconvOnceByKernel3RowTwo(scheme,image,res,doubleWeight,start+3,start+5,globalSizeCol,3);
        multiconvOnceByKernel3RowThree(scheme,image,res,doubleWeight,start+6,start+8,globalSizeCol,6);      
        return res;
    }

    void multiconvOnceByKernel3RowOne(Scheme & scheme, vector<Ciphertext> &image, vector<Ciphertext> &res, vector<double>&doubleWeight,int start,int end,int cols,int base){
        for (int i = start; i <= end; i++)
        {
            vector<Ciphertext> tmpRotate(image);
            if(cipherNum == 1){
                vector<int> moveSteps = getMoveSteps(cols+1-(i-start));
                for (int o = 0; o < moveSteps.size(); o++){
                    scheme.rightRotateFastAndEqual(tmpRotate[0],moveSteps[o]);
                }
            }
            else
                multiRightRot(scheme, tmpRotate, cipherNum, slots, cols+1-(i-start));

            vector<cdouble *> complexWeight = multigetConvWeightArrange(doubleWeight[i],i-start+base,slots);
            for(int j=0; j<cipherNum; j++){
                Ciphertext tmp(tmpRotate[j]);
                scheme.multByConstVecAndEqual(tmp,complexWeight[j],logp);
                scheme.reScaleByAndEqual(tmp,logp);
                scheme.modDownToAndEqual(res[j],tmp.logq);
                scheme.addAndEqual(res[j],tmp);
                delete []complexWeight[j];
            }
        }

    }

    void multiconvOnceByKernel3RowTwo(Scheme & scheme, vector<Ciphertext> &image, vector<Ciphertext> &res, vector<double>&doubleWeight,int start,int end,int cols,int base){

        for (int i = start; i <= end; i++)
        {
            vector<Ciphertext> tmpRotate(image);
            if(i == start){
                if(cipherNum == 1){
                    scheme.rightRotateFastAndEqual(tmpRotate[0],1);
                }
                else
                    multiRightRot(scheme, tmpRotate, cipherNum, slots, 1);
            }
            if(i == end){
                if(cipherNum == 1){
                    scheme.leftRotateFastAndEqual(tmpRotate[0],1);
                }
                else
                    multiLeftRot(scheme, tmpRotate, cipherNum, slots, 1);
            }

            vector<cdouble *> complexWeight = multigetConvWeightArrange(doubleWeight[i],i-start+base,slots);
            for(int j=0; j<cipherNum; j++){
                Ciphertext tmp(tmpRotate[j]);
                scheme.multByConstVecAndEqual(tmp,complexWeight[j],logp);
                scheme.reScaleByAndEqual(tmp,logp);
                scheme.modDownToAndEqual(res[j],tmp.logq);
                scheme.addAndEqual(res[j],tmp);
                delete []complexWeight[j];
            }
        }
    }

    void multiconvOnceByKernel3RowThree(Scheme & scheme, vector<Ciphertext> &image, vector<Ciphertext> &res, vector<double>&doubleWeight,int start,int end,int cols,int base){
        
        for (int i = start; i <= end; i++)
        {
            vector<Ciphertext> tmpRotate(image);
            if(cipherNum == 1){
                vector<int> moveSteps = getMoveSteps(cols-1+(i-start));
                for (int o = 0; o < moveSteps.size(); o++){
                    scheme.leftRotateFastAndEqual(tmpRotate[0],moveSteps[o]);
                }
            }
            else
                multiLeftRot(scheme, tmpRotate, cipherNum, slots, cols-1+(i-start));

            vector<cdouble *> complexWeight = multigetConvWeightArrange(doubleWeight[i],i-start+base,slots);
            for(int j=0; j<cipherNum; j++){
                Ciphertext tmp(tmpRotate[j]);
                scheme.multByConstVecAndEqual(tmp,complexWeight[j],logp);
                scheme.reScaleByAndEqual(tmp,logp);
                scheme.modDownToAndEqual(res[j],tmp.logq);
                scheme.addAndEqual(res[j],tmp);
                delete []complexWeight[j];
            }
        }
    }

    vector<cdouble *> multigetConvWeightArrange(double weight,int index,int slots){
        vector<cdouble *> res(cipherNum);

        for (int k=0; k<cipherNum; k++){
            cdouble * tmp = new cdouble[slots];
            for(int i = 0; i < slots; i++){
                tmp[i] = multiglobalWeightMasks[index][k][i] *weight;
            }
            res[k] = tmp;
        }
        return res;
    }

    // Multi-ciphertext activation
    void multirelu(vector<vector<Ciphertext>> &image, int cipherNum){
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < image.size(); i++)
        {   
            for(int j=0; j<cipherNum; j++){
                Ciphertext secondLevelCipher,thirdLevelCipher;
                scheme.mult(secondLevelCipher,image[i][j],image[i][j]);
                scheme.reScaleByAndEqual(secondLevelCipher,logp);

                scheme.multByConst(thirdLevelCipher,image[i][j],actParams[3],logp);
                scheme.reScaleByAndEqual(thirdLevelCipher,logp);
                scheme.multAndEqual(thirdLevelCipher,secondLevelCipher);
                scheme.reScaleByAndEqual(thirdLevelCipher,logp);

                scheme.multByConstAndEqual(secondLevelCipher,actParams[2],logp);
                scheme.reScaleByAndEqual(secondLevelCipher,logp);

                scheme.multByConst(image[i][j],image[i][j],actParams[1],logp);
                scheme.reScaleByAndEqual(image[i][j],logp);

                scheme.addConstAndEqual(image[i][j],actParams[0],logp);
                scheme.addAndEqual(secondLevelCipher,thirdLevelCipher);

                scheme.modDownToAndEqual(image[i][j], secondLevelCipher.logq );
                scheme.add(image[i][j],image[i][j],secondLevelCipher);
            }
        }
        cout << getCurrentRSS()<<"GB" <<endl;
    }

    // Multi-ciphertext 1x1 convolution
    vector<vector<Ciphertext>> multiconv1x1(vector<vector<Ciphertext>> image, int level, int cipherNum){
        vector<vector<double> >weights(outChanels[level],vector<double>(inChanels[level],0));
        loadParams(weights,level);

        vector<vector<cdouble>> complexweights(outChanels[level], vector<cdouble>(inChanels[level]));
        complexweights = doubleToComplex(weights);

        vector<vector<Ciphertext>> res(outChanels[level], vector<Ciphertext>(cipherNum));
        for (int i = 0; i < res.size(); i++){
            for (int k=0; k<cipherNum; k++){
                scheme.encryptZeros(res[i][k],slots,logp,logq);
            }
        }

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < outChanels[level]; i++)
        {
            vector<Ciphertext> cipher(cipherNum);
            for (int o=0; o<cipherNum; o++){
                scheme.encryptZeros(cipher[o],slots,logp,logq);
            }
            
            for (int j = 0; j < inChanels[level]; j++)
            {          
                vector<Ciphertext> tmp(cipherNum);
                for (int k=0; k<cipherNum; k++){
                    scheme.multByConst(tmp[k],image[j][k],complexweights[i][j],logp);
                    scheme.reScaleByAndEqual(tmp[k],logp);
                    scheme.modDownToAndEqual(cipher[k],tmp[k].logq);
                    scheme.addAndEqual(cipher[k],tmp[k]);
                }
            }

            for (int u=0; u<cipherNum; u++){
                scheme.modDownToAndEqual(res[i][u],cipher[u].logq);
                scheme.addAndEqual(res[i][u],cipher[u]);
            }
        }
        cout << getCurrentRSS()<<"GB" <<endl;
        return res;
    }

    // Multi-ciphertext sub-pixel convolution
    vector<vector<Ciphertext>> multipixel(vector<vector<Ciphertext>> image, int scale, int cipherNum, int slots){
        int pos, channe, channnum, newcipherNum, cino, finpos, finno;
        newcipherNum = ceil(((double)sizeRow*scale*sizeCol*scale)/((double)slots)); 
        channnum = image.size() / (scale * scale);
        vector<vector<Ciphertext>> res(channnum,vector<Ciphertext>(newcipherNum));
    	for (int i = 0; i < channnum; i++){
            for (int k=0; k<newcipherNum; k++){
                scheme.encryptZeros(res[i][k],slots,logp,logq);
               
            }
        }
        vector<int> moveSteps;
        for(int k=0; k < channnum; k++){
            for(int i = 0; i < sizeRow*scale; i++){
                for(int j = 0; j < sizeCol*scale; j++){            
                    pos = sizeCol * (i/scale) + (j/scale);
                    channe = k * scale * scale + (i % scale) * scale + (j % scale);
                    cino = floor(pos/slots);
                    finpos = pos % slots;
                    moveSteps = my_getMoveSteps(finpos);
                    Ciphertext tmp = computepixel(scheme, image[channe][cino], moveSteps, finpos);
                    scheme.modDownToAndEqual(res[k][finno], tmp.logq);
                    scheme.addAndEqual(res[k][finno], tmp);
                    scheme.leftRotateFastAndEqual(res[k][finno], 1);             
                }
            }
        }
        int last;
        int mark = 0;
        last = (sizeRow*scale*sizeCol*scale) % slots;
        if(last > slots/2){
            last = slots - last;
            mark = 1;
        }
        moveSteps = my_getMoveSteps(last);
        if(mark == 0){
            for(int i = 0; i < channnum; i++){
                for(int j = 0; j< newcipherNum; j++){
                    for (int l = 0; l < moveSteps.size(); l++){
                        scheme.rightRotateFastAndEqual(res[i][j], moveSteps[l]);
                    }
                }
            }
        }
        else if(mark == 1){
            for(int i = 0; i < channnum; i++){
                for(int j = 0; j< newcipherNum; j++){
                    for (int l = 0; l < moveSteps.size(); l++){
                        scheme.leftRotateFastAndEqual(res[i][j], moveSteps[l]);
                    }
                }
            }
        }    
        return res;
    }
    
   
///////////// Improved bicubic interpolation [single ciphertext version] ///////////////////////////////////////////////////
    Ciphertext BicConv(Ciphertext &image, vector<double> doubleWeight){
        Ciphertext res;
    	scheme.encryptZeros(res, slots, logp, logq);

        int rotstep;
        for(int i=0; i<2*fa; i++){
            rotstep = (2*fa-1-i) * globalSizeCol + (2*fa-1);
            RightBicConv(scheme,image,res,doubleWeight,i*BickernelCol,rotstep);
        }

        for(int i=2*fa; i<4*fa; i++){
            rotstep = (i-2*fa+1) * globalSizeCol - (2*fa-1);
            leftBicConv(scheme,image,res,doubleWeight,i*BickernelCol,rotstep);
        }

        return res;
    }

    void RightBicConv(Scheme & scheme, Ciphertext &image, Ciphertext &res, vector<double> doubleWeight,int start,int rotstep){
        Ciphertext tmpRotate(image);
        
        vector<int> moveSteps = getMoveSteps(rotstep);
        for (int i = 0; i < moveSteps.size(); i++)
        {
            scheme.rightRotateFastAndEqual(tmpRotate,moveSteps[i]);
        }
        
        
        for (int i = 0; i < BickernelCol; i++)
        {
            // Get the corresponding plaintext weight packing
            cdouble * complexWeight = getBicConvWeightArrange(doubleWeight[i+start],i+start,slots);
            Ciphertext tmp(tmpRotate);
            scheme.multByConstVecAndEqual(tmp,complexWeight,logp);
            scheme.reScaleByAndEqual(tmp,logp);
            scheme.modDownToAndEqual(res,tmp.logq);
            scheme.addAndEqual(res,tmp);

            delete []complexWeight;
            // Left shift
            scheme.leftRotateFastAndEqual(tmpRotate,1);
        }
    }

    void leftBicConv(Scheme & scheme, Ciphertext &image, Ciphertext &res, vector<double> doubleWeight,int start,int rotstep){
        Ciphertext tmpRotate(image);
        
        vector<int> moveSteps = getMoveSteps(rotstep);
        for (int i = 0; i < moveSteps.size(); i++)
        {
            scheme.leftRotateFastAndEqual(tmpRotate,moveSteps[i]);
        }
        
        
        for (int i = 0; i < BickernelCol; i++)
        {
            // Get the corresponding plaintext weight packing
            cdouble * complexWeight = getBicConvWeightArrange(doubleWeight[i+start],i+start,slots);
            Ciphertext tmp(tmpRotate);
            scheme.multByConstVecAndEqual(tmp,complexWeight,logp);
            scheme.reScaleByAndEqual(tmp,logp);
            scheme.modDownToAndEqual(res,tmp.logq);
            scheme.addAndEqual(res,tmp);

            delete []complexWeight;
            // Left shift
            scheme.leftRotateFastAndEqual(tmpRotate,1);
        }
    }

    cdouble* getBicConvWeightArrange(double weight,int index,int slots){
        cdouble * res = new cdouble[slots];
        for (int i = 0; i < slots; i++)
        {
            res[i] = BicWegihtMasks[index][i] * weight;
        }
        return res;
    }


/////////////   Bicubic interpolation (normal version)   //////////////////////////////////////////////
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
//         res = 0;
//     return res;
// }

    double abs(double x){
        double res;
        if (x <= 0){
            res =  -x;
        }
        else{
            res =  x;
        }
        return res;
    }

    void getW(double w_x[4], double x){
        double a = -0.5;
        int X = (int)x;
        double stemp_x[4];
        stemp_x[0] = 1 + (x - X);
        stemp_x[1] = x - X;
        stemp_x[2] = 1 - (x - X);
        stemp_x[3] = 2 - (x - X);

        w_x[0] = a*abs(stemp_x[0] * stemp_x[0] * stemp_x[0]) - 5 * a*stemp_x[0] * stemp_x[0] + 8 * a*abs(stemp_x[0]) - 4 * a;
        w_x[1] = (a + 2)*abs(stemp_x[1] * stemp_x[1] * stemp_x[1]) - (a + 3)*stemp_x[1] * stemp_x[1] + 1;
        w_x[2] = (a + 2)*abs(stemp_x[2] * stemp_x[2] * stemp_x[2]) - (a + 3)*stemp_x[2] * stemp_x[2] + 1;
        w_x[3] = a*abs(stemp_x[3] * stemp_x[3] * stemp_x[3]) - 5 * a*stemp_x[3] * stemp_x[3] + 8 * a*abs(stemp_x[3]) - 4 * a;
    }


    vector<vector<Ciphertext>> Bicu(Scheme & scheme, vector<vector<Ciphertext>> image, int scale, int sizeRow, int sizeCol, int slots){
        Ciphertext zero;
        scheme.encryptZeros(zero,slots,logp,logq);
        vector<vector<Ciphertext>> res(inputchannel,vector<Ciphertext>(cipherNum,zero));
        // vector<vector<Ciphertext>> restmp(sizeRow, vector<Ciphertext>(sizeCol));
        int initr,initc;
        double w_i[4], w_j[4]; 
        double u,v;          
        initr = (int)sizeRow / scale;
        initc = (int)sizeCol / scale;
        int initpos, cipherpos, pos, initpos1, cipherpos1;
        int k, i, j, m, n, g;
        double x, y;
        int x1, y1;
        for(k=0; k<inputchannel; k++){
            for(i=sizeRow-1; i>=0; i--){
                for(j=sizeCol-1; j>=0; j--){
                    cout << i << j << endl;
                    x = i * initr/sizeRow;
                    y = j * initc/sizeCol;
                    x1 = (int)x;
                    y1 = (int)y;
                    getW(w_i, x);
                    getW(w_j, y);
                    Ciphertext sum(zero);
                    for(m = 0; m <= 3; m++){
                        for(n = 0; n <= 3; n++){
                            if ((x1+m-1>=0) && (y1+n-1>=0) && (x1+m-1<initr) && (y1+n-1<initc)){
                                cdouble * w = new cdouble[slots];
                                initpos = (x1+m-1) * initc +(y1+n-1);
                                cipherpos = floor(initpos/slots);
                                pos = initpos % slots;
                                w[pos] = cdouble(1,0) * w_i[m] * w_j[n];
                                Ciphertext tmp(image[k][cipherpos]);
                                scheme.multByConstVecAndEqual(tmp, w, logp);
                                scheme.reScaleByAndEqual(tmp, logp);
                                vector<int> moveSteps = getMoveSteps(pos);
                                for (g = 0; g < moveSteps.size(); g++){
                                    scheme.leftRotateFastAndEqual(tmp,moveSteps[g]);
                                }
                                scheme.modDownToAndEqual(sum, tmp.logq);
                                scheme.addAndEqual(sum, tmp);
                                delete []w;
                            }  
                        }
                    }
                    initpos1 = i * sizeCol +j;
                    cipherpos1 = floor(initpos/slots);
                    scheme.modDownToAndEqual(res[k][cipherpos1], sum.logq);
                    scheme.addAndEqual(res[k][cipherpos1], sum);
                    scheme.rightRotateFastAndEqual(res[k][cipherpos1],1);
                }
            }
        }
        cout << getCurrentRSS()<<"GB" <<endl;
        return res;
    }



/////////////   Plaintext domain scheme   //////////////////////////////////////////////
    vector<Ciphertext> my_conv(vector<Ciphertext> image, int level){
        vector<vector<double> >weights(outChanels[level],vector<double>(inChanels[level]*kernelSize*kernelSize,0));
        for(int i=0; i<outChanels[level]; i++){
            for(int j=0; j<inChanels[level]*kernelSize*kernelSize; j++){
                weights[i][j] = 1;
            }
        }

        vector<Ciphertext> res(outChanels[level]);
        for (int i = 0; i < res.size(); i++)
        {
            scheme.encryptZeros(res[i],slots,logp,logq);
        }

        int gapLen = kernelSize * kernelSize;

        // #pragma omp parallel for schedule(static)
        timeutils.start("inconv...");
        for(int i = 0; i < outChanels[level]; i++)
        {
            // Initialize to zero
            Ciphertext cipher;
            scheme.encryptZeros(cipher, slots, logp, logq);

            for(int j = 0; j < inChanels[level]; j++)
            {
                int s;
                s = sizeCol-kernelSize +1;
                for(int m = 0; m < s*s; m++){    
                    Ciphertext tmp = my_convOnceByKernel3(image[0],weights[i],j*gapLen,(j+1)*gapLen-1);// Function modified here
                    scheme.modDownToAndEqual(cipher,tmp.logq);
                    scheme.addAndEqual(cipher,tmp); 
                }
                
            }
            scheme.modDownToAndEqual(res[i],cipher.logq);
            scheme.addAndEqual(res[i],cipher);
        }
        timeutils.stop("inconv...");
        cout << getCurrentRSS()<<"GB" <<endl;
        
        return res;
    }

    // Convolution with 3*3 kernel
    Ciphertext my_convOnceByKernel3(Ciphertext &image, vector<double>&doubleWeight,int start,int end){
        Ciphertext res;
    	scheme.encryptZeros(res, slots, logp, logq);

        my_convOnceByKernel3RowOne(scheme,image,res,doubleWeight,start,start+2,globalSizeCol,0);// Only function parameters modified here
        // cout << "RowOne OK..\n";
        my_convOnceByKernel3RowTwo(scheme,image,res,doubleWeight,start+3,start+5,globalSizeCol,3);// Only function parameters modified here
        // cout << "RowTwo OK..\n";
        my_convOnceByKernel3RowThree(scheme,image,res,doubleWeight,start+6,start+8,globalSizeCol,6);// Only function parameters modified here
        // cout << "RowThree OK..\n";

        
        return res;
    }

    // First row of 3*3 convolution
    void my_convOnceByKernel3RowOne(Scheme & scheme, Ciphertext &image, Ciphertext &res, vector<double>&doubleWeight,int start,int end,int cols,int base){
        Ciphertext tmpRotate(image);
        // Need to move col+1 steps, but not necessarily a power of 2, need to decompose
        // vector<int> moveSteps = getMoveSteps(cols+1);
        // for (int i = 0; i < moveSteps.size(); i++)
        // {
        //     scheme.rightRotateFastAndEqual(tmpRotate,moveSteps[i]);
        // }
        

        for (int i = start; i <= end; i++)
        {
            // Get the corresponding plaintext weight packing
            cdouble * complexWeight = new cdouble[slots];
            complexWeight[0] = cdouble(1,0);
            
            Ciphertext tmp(tmpRotate);
            scheme.multByConstVecAndEqual(tmp,complexWeight,logp);
            scheme.reScaleByAndEqual(tmp,logp);
            scheme.modDownToAndEqual(res,tmp.logq);
            scheme.addAndEqual(res,tmp);

            delete []complexWeight;
            // cout << "delete OK\n";
            // Left shift
            // scheme.leftRotateFastAndEqual(tmpRotate,1);
            int a=1;
            // cout << "leftRotateFastAndEqual OK\n";
        }
    }

    // Second row of 3*3 convolution
    void my_convOnceByKernel3RowTwo(Scheme & scheme, Ciphertext &image, Ciphertext &res, vector<double>&doubleWeight,int start,int end,int cols,int base){
        Ciphertext tmpRotate(image);

        // scheme.rightRotateFastAndEqual(tmpRotate,1);
        int a=1;
        
        
        for (int i = start; i <= end; i++)
        {
            // Get the corresponding plaintext weight packing
            cdouble * complexWeight = new cdouble[slots];
            complexWeight[0] = cdouble(1,0);
            Ciphertext tmp(tmpRotate);
            scheme.multByConstVecAndEqual(tmp,complexWeight,logp);
            scheme.reScaleByAndEqual(tmp,logp);
            scheme.modDownToAndEqual(res,tmp.logq);
            scheme.addAndEqual(res,tmp);

            delete []complexWeight;
            // Left shift
            // scheme.leftRotateFastAndEqual(tmpRotate,1);
            int a=1;
        }
    }

    // Third row of 3*3 convolution
    void my_convOnceByKernel3RowThree(Scheme & scheme, Ciphertext &image, Ciphertext &res, vector<double>&doubleWeight,int start,int end,int cols,int base){
        Ciphertext tmpRotate(image);
        
        // Need to move col-1 steps, but not necessarily a power of 2, need to decompose
        // vector<int> moveSteps = getMoveSteps(cols-1);
        // for (int i = 0; i < moveSteps.size(); i++)
        // {
        //     scheme.leftRotateFastAndEqual(tmpRotate,moveSteps[i]);
        // }
        
        
        for (int i = start; i <= end; i++)
        {
            // Get the corresponding plaintext weight packing
            cdouble * complexWeight = new cdouble[slots];
            complexWeight[0] = cdouble(1,0);
            Ciphertext tmp(tmpRotate);
            scheme.multByConstVecAndEqual(tmp,complexWeight,logp);
            scheme.reScaleByAndEqual(tmp,logp);
            scheme.modDownToAndEqual(res,tmp.logq);
            scheme.addAndEqual(res,tmp);

            delete []complexWeight;
            // Left shift
            // scheme.leftRotateFastAndEqual(tmpRotate,1);
            int a=1;
        }
    }


    // Activation function
    vector<Ciphertext> my_relu(vector<Ciphertext> &image){
        // 6.83593757e-06*(tri(x))+2.34375001e-01*(square(x))+4.99988281e-01*x+1.87499999e-01
        Ciphertext zero;
        scheme.encryptZeros(zero, slots, logp, logq);
        vector<Ciphertext> res(sizeCol*sizeRow, zero);
        // vector<Ciphertext> tmpCipher(image);
        // Process each ciphertext in image
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < 1; i++)
        {   
            cout << i << endl;
            for(int j=0; j<sizeCol*sizeRow; j++){
                vector<Ciphertext> tmpCipher(image);
                Ciphertext secondLevelCipher,thirdLevelCipher,tmp;
                scheme.mult(secondLevelCipher,tmpCipher[0],tmpCipher[0]);
                scheme.reScaleByAndEqual(secondLevelCipher,logp);

                // Get cubic ciphertext (optimized to reduce multiplication depth)
                scheme.multByConst(thirdLevelCipher,tmpCipher[0],actParams[3],logp);
                scheme.reScaleByAndEqual(thirdLevelCipher,logp);
                scheme.multAndEqual(thirdLevelCipher,secondLevelCipher);
                scheme.reScaleByAndEqual(thirdLevelCipher,logp);

                // Get quadratic ciphertext
                scheme.multByConstAndEqual(secondLevelCipher,actParams[2],logp);
                scheme.reScaleByAndEqual(secondLevelCipher,logp);

                // Get linear ciphertext
                scheme.multByConst(tmp,tmpCipher[0],actParams[1],logp);
                scheme.reScaleByAndEqual(tmp,logp);

                // Add constant term to linear term, add quadratic and cubic terms
                scheme.addConstAndEqual(tmp,actParams[0],logp);
                scheme.addAndEqual(secondLevelCipher,thirdLevelCipher);

                // Adjust logq of firstLevelCipher
                scheme.modDownToAndEqual(tmp, secondLevelCipher.logq );

                // Add both together
                scheme.add(tmp,tmp,secondLevelCipher);

                scheme.modDownToAndEqual(res[i],tmp.logq);
                scheme.addAndEqual(res[i],tmp);
            }
        }
        return res;
    }

     vector<Ciphertext> my_residual(vector<Ciphertext> &image, vector<Ciphertext> image1){
        
        Ciphertext zero;
        scheme.encryptZeros(zero, slots, logp, logq);
        vector<Ciphertext> res(sizeCol*sizeRow, zero);
        for(int j=0; j<16; j++){
            for(int i=0; i<sizeCol*sizeRow; i++){
                Ciphertext tmp(zero);
                scheme.modDownToAndEqual(tmp,image[0].logq);
                scheme.addAndEqual(tmp,image[0]);

                scheme.modDownToAndEqual(tmp,image1[0].logq);
                scheme.addAndEqual(tmp,image1[0]);

                scheme.modDownToAndEqual(res[0],tmp.logq);
                scheme.addAndEqual(res[0],tmp);
            }
        }
        return res;
     }
/////////////   Plaintext domain scheme   //////////////////////////////////////////////

private:
	Scheme& scheme;
    SecretKey &secretKey;
    vector<cdouble>actParams;
    int sizeRow;
    int sizeCol;

public:
    cipherNet(Scheme& scheme_,vector<double>&activationParams,int sizeRow,int sizeCol,SecretKey &secretKey_):scheme(scheme_),secretKey(secretKey_){
        // Reverse parameters so that the lowest bit corresponds to constant, highest bit corresponds to coefficient at position with exponent 3
        reverse(activationParams.begin(),activationParams.end());
        actParams = doubleToComplexVector(activationParams);
        this->sizeCol = sizeCol;
        this->sizeRow = sizeRow;
    }
};

#endif