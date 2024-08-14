
/*
 * Sorting algorithm using the CKKS scheme.
 * note: initial level should be set carefully to minimize bootstapping
*/

#include "openfhe.h"
#include <cstdio>
#include <random>
#include <algorithm>
using namespace lbcrypto;
using Ctxt = Ciphertext<DCRTPoly>;
using Ptxt = Plaintext;
using context = const CryptoContext<DCRTPoly>;
bool DEBUG = true;
#define trueLevel(ciphertextAfter) (ciphertextAfter->GetLevel() + (ciphertextAfter->GetNoiseScaleDeg() - 1))

// 1. You should run this code and observe the level consupmtion at each while loop
// 2. You should then modify the level budget and the condition for bootstrapping 
// 3. note that changing polydegree in compare_and_swap may influence the level consumption

Ctxt compare_and_swap(context& cc,Ctxt& a1,Ctxt& a2,Ctxt& a3,Ctxt& a4){
    Ctxt a1_sub_a2 = cc->EvalSub(a1,a2);
    Ctxt a2_sub_a1 = cc->EvalSub(a2,a1);
    double lowerBound = -5;
    double upperBound = 5;
    int polyDegree = 119;
    auto a1_gt_a2 = cc->EvalChebyshevFunction([](double x) -> double
                                              { if(x>=0) return 1;else return 0; },
                                              a1_sub_a2, lowerBound,
                                              upperBound, polyDegree);
    auto a2_gt_a1 = cc->EvalChebyshevFunction([](double x) -> double
                                                { if(x>0) return 1;else return 0;},
                                                a2_sub_a1, lowerBound,
                                                upperBound, polyDegree);

    return cc->EvalAdd(cc->EvalMult(a1_gt_a2,a3),cc->EvalMult(a2_gt_a1,a4));
    }

void Sorting(size_t input_length = 8);

int main(int argc, char *argv[])
{
    Sorting(1<<14);
    return 0;
}

void Sorting(size_t input_length) {
    std::cout << "--------------------------------- Sorting ---------------------------------"
              << std::endl;
    size_t total_steps=(1+log2(input_length))*log2(input_length)/2;

    std::cout<<"Total steps: "<<total_steps<<"\n";
    std::vector<double> input_msg(input_length,0);
    size_t encodedLength = input_length;
    std::random_device rd; // Obtain a random number from hardware
    std::mt19937 gen(rd()); // Seed the generator
    std::uniform_real_distribution<> distr(3,4); // Define the range

    for(size_t i = 0; i < input_length; ++i) {
        input_msg[i] = distr(gen); // Generate a random number within the range
    }

    // std::cout<<"Generated input vector: "<<input_msg<<"\n";
    // Selecting CKKS parameters
    CCParams<CryptoContextCKKSRNS> parameters;
    size_t ringDimension = 1 << 17;
    size_t numSlots = input_length;
    size_t dnum = 3;
    SecretKeyDist secretKeyDist = SPARSE_TERNARY;
    std::vector<uint32_t> levelBudget = {2, 2};
    uint32_t levelsAvailableAfterBootstrap = 34;
    uint32_t multDepth = levelsAvailableAfterBootstrap + FHECKKSRNS::GetBootstrapDepth(levelBudget, secretKeyDist);
    ScalingTechnique rescaleTech = FLEXIBLEAUTOEXT;
    std::cout<<"multDepth: "<<multDepth<<"\n";
    // int numSlots = 1 << 9;
    parameters.SetBatchSize(numSlots);
    parameters.SetNumLargeDigits(dnum);
#if NATIVEINT == 128
    usint scalingModSize = 78;
    usint firstModSize   = 89;
#else
    usint scalingModSize = 59;
    usint firstModSize   = 60;
#endif
    parameters.SetSecretKeyDist(secretKeyDist);
    parameters.SetSecurityLevel(HEStd_128_classic);
    parameters.SetRingDim(ringDimension);
    parameters.SetScalingModSize(scalingModSize);
    parameters.SetFirstModSize(firstModSize);
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingTechnique(rescaleTech);
    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(FHE);

    cc->EvalBootstrapSetup(levelBudget,{0, 0},numSlots);

    auto keyPair = cc->KeyGen();
    cc->EvalMultKeyGen(keyPair.secretKey);
    cc->EvalBootstrapKeyGen(keyPair.secretKey, numSlots);
    std::cout<<"mult_key generation finished\n";
    std::vector<int> rotate_keygen_list;
    // 1,2,4,8,16,32,64,128,256,-1,-2,-4,-8,-16,-32,-64,-128,-256
    for (size_t i = 1; i < input_length; i <<= 1) {
      rotate_keygen_list.push_back(i);
      rotate_keygen_list.push_back(-i);
    }

    cc->EvalRotateKeyGen(keyPair.secretKey, rotate_keygen_list, keyPair.publicKey);
    std::cout<<"rotate_key generation finished\n";

    // Input preparation
    Plaintext input_pt  = cc->MakeCKKSPackedPlaintext(input_msg);
    // Plaintext input_pt  = cc->MakeCKKSPackedPlaintext(input_msg);
    std::cout << "depth: " << multDepth << std::endl;
    auto input_ct      = cc->Encrypt(keyPair.publicKey, input_pt);
    std::cout << "Initial number of levels remaining: " << multDepth - input_ct->GetLevel() << std::endl;

    // Sorting
    size_t n = encodedLength;
    size_t k = 2;
    size_t step = 1;
    Plaintext plaintextDec;
    std::sort(input_msg.begin(), input_msg.end(), std::less<double>());
    while (k <= n)
    {
        size_t j = k / 2;
        while(j>0){
            if(multDepth-trueLevel(input_ct) <=10){
                std::cout<<" level before bootstrapping:  "<<multDepth-trueLevel(input_ct)<<"\n";
                input_ct = cc->EvalBootstrap(input_ct, 2, 20);
                std::cout<<" level after bootstrapping:  "<<multDepth-trueLevel(input_ct)<<"\n";
            }

            std::cout<<"step: "<<step<<"\n";
            std::cerr << "[APP TRACE] step: " << step << "(k=" << k << ",j=" << j << ")" << std::endl;
            step++;
            std::vector<double> mask1(n,0);
            std::vector<double> mask2(n,0);
            std::vector<double> mask3(n,0);
            std::vector<double> mask4(n,0);

            for (size_t i = 0; i < n;i++){
                size_t l = i ^ j;
                if(i < l){
                    if((i&k) == 0){
                        mask1[i] = 1;
                        mask2[l] = 1;
                    }else{
                        mask3[i] = 1;
                        mask4[l] = 1;
                    }
                }
            }
            Ctxt arr1 = cc->EvalMult(input_ct,cc->MakeCKKSPackedPlaintext(mask1));
            Ctxt arr2 = cc->EvalMult(input_ct,cc->MakeCKKSPackedPlaintext(mask2));
            Ctxt arr3 = cc->EvalMult(input_ct,cc->MakeCKKSPackedPlaintext(mask3));
            Ctxt arr4 = cc->EvalMult(input_ct,cc->MakeCKKSPackedPlaintext(mask4));
            Ctxt arr5_1 = cc->EvalRotate(arr1,-j);
            Ctxt arr5_2 = cc->EvalRotate(arr3,-j);
            Ctxt arr6_1 = cc->EvalRotate(arr2,j);
            Ctxt arr6_2 = cc->EvalRotate(arr4,j);
            Ctxt arr7 = cc->EvalAdd(cc->EvalAdd(arr5_1,arr5_2),cc->EvalAdd(arr6_1,arr6_2));
            Ctxt arr8 = input_ct;
            Ctxt arr9 = cc->EvalAdd(cc->EvalAdd(arr5_1,arr1),cc->EvalAdd(arr6_2,arr4));
            Ctxt arr10 = cc->EvalAdd(cc->EvalAdd(arr5_2,arr3),cc->EvalAdd(arr6_1,arr2));

            input_ct = compare_and_swap(cc,arr7,arr8,arr9,arr10);
            j = j / 2;

            std::cout<<"remaining level:  "<<multDepth-trueLevel(input_ct)<<"\n";
            if (DEBUG) {
            cc->Decrypt(keyPair.secretKey, input_ct, &plaintextDec);
            plaintextDec->SetLength(encodedLength);
            auto tmp_result = plaintextDec->GetRealPackedValue();
            std::sort(tmp_result.begin(), tmp_result.end(), std::less<double>());
            double total_error = 0;
            for (size_t i = 0; i < encodedLength; i++) {
              total_error += (tmp_result[i]-input_msg[i]) * (tmp_result[i]-input_msg[i]);
            }
           std::cout<<"Avg error: "<< total_error/encodedLength << std::endl;
            }
        }
    k *= 2;
    // Level consumption: ~12 level
    }
    std::cerr << "[APP TRACE] finish" << std::endl;
    // Output computation
    cc->Decrypt(keyPair.secretKey, input_ct, &plaintextDec);
    plaintextDec->SetLength(encodedLength);
    
    
    // std::sort(input_msg.begin(), input_msg.end(), std::less<double>());
    // std::cout<<"Expected output: "<< input_msg << std::endl;

    std::vector<std::complex<double>> finalResult = plaintextDec->GetCKKSPackedValue();
    std::vector<double> tmp_result = plaintextDec->GetRealPackedValue();
    // std::cout << "Actual output: " << finalResult << std::endl << std::endl;
    
    std::vector<double> differences;
    std::vector<double> magnitudes;
    std::vector<double> ratio_avg;

    for (size_t i = 0; i < input_msg.size(); ++i)
    {
        double magnitude = std::abs(finalResult[i]);
        magnitudes.push_back(magnitude);
        differences.push_back(std::abs(input_msg[i] - magnitude));
        ratio_avg.push_back(differences[i] / magnitude);
    }

    // Compute the average difference
    double avgDifference = std::accumulate(differences.begin(), differences.end(), 0.0) / differences.size();

    // Compute the average magnitude of the complex numbers
    double avgMagnitude = std::accumulate(magnitudes.begin(), magnitudes.end(), 0.0) / magnitudes.size();

    double ratioavg = std::accumulate(ratio_avg.begin(), ratio_avg.end(), 0.0) / ratio_avg.size();
    // Compute the average difference divided by the average magnitude
    double ratio = avgDifference / avgMagnitude;
    FILE* fp = fopen("input_msg.bin", "wb");
    fwrite(input_msg.data(), sizeof(double), input_msg.size(), fp);
    fclose(fp);
    fp = fopen("output.bin", "wb");
    fwrite(tmp_result.data(), sizeof(double), tmp_result.size(), fp);
    fclose(fp);

    std::cout << "Average difference: " << avgDifference << std::endl;
    std::cout << "Average magnitude of complex numbers: " << avgMagnitude << std::endl;
    std::cout << "Average difference / Average magnitude: " << ratio << std::endl;
    std::cout << "Average (difference / magnitude): " << ratioavg << std::endl;

}