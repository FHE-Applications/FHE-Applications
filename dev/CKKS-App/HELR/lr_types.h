//==================================================================================
// BSD 2-Clause License
//
// Copyright (c) 2023, Duality Technologies Inc.
//
// All rights reserved.
//
// Author TPOC: contact@openfhe.org
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//==================================================================================

#ifndef DPRIVE_ML__LR_TYPES_H_
#define DPRIVE_ML__LR_TYPES_H_

#include "openfhe.h"

typedef std::numeric_limits<double> dbl;

//todo replace typedef with using =
typedef double prim_type; //do we really need this still? Ideally it is so code works with other POD types
typedef std::vector<prim_type> Vec;
typedef std::vector<std::vector<prim_type>> Mat;

using CC = lbcrypto::CryptoContext<lbcrypto::DCRTPoly>; //crypto contexts
using CT = lbcrypto::Ciphertext<lbcrypto::DCRTPoly>; //ciphertext
using PT = lbcrypto::Plaintext; //plaintext
using CryptoParams = lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS>;

using KeyPair = lbcrypto::KeyPair<lbcrypto::DCRTPoly>;
using MatKeys = std::shared_ptr<std::map<usint, lbcrypto::EvalKey < lbcrypto::DCRTPoly>>>;

#endif //DPRIVE_ML__LR_TYPES_H_
