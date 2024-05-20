// Copyright (C) 2018  Philipp Basler and Margarete Mühlleitner
// SPDX-FileCopyrightText: 2021 Philipp Basler, Margarete Mühlleitner and Jonas
// Müller
//
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * @file
 */

#pragma once

#include <string> // for string
#include <vector> // for vector

#include <BSMPT/models/ClassPotentialOrigin.h>
namespace BSMPT
{
namespace Models
{

/**
 * @brief The Class_VDM class
 * Template for implementing a new model
 */
class Class_VDM : public Class_Potential_Origin
{
public:
  Class_VDM(const ISMConstants &smConstants);
  virtual ~Class_VDM();

  // Add here your parameters for the Lagrangian as well as for the counterterm
  // potential Add here your variables in which you will save the Debye
  // correction factors

  double muHSq, muSSq, lambdaH, lambdaS, kappa;
  
  double MH1, MH2, alpha, MX;

  double dmuHSq, dmuSSq, dlambdaH, dlambdaS, dkappa, dT1, dT2,
      dT3, dT4, dT5, dT6;
      
  double v, vs;
  
  double gX;
  
  


  void ReadAndSet(const std::string &linestr,
                  std::vector<double> &par) override;
  std::vector<std::string> addLegendCT() const override;
  std::vector<std::string> addLegendTemp() const override;
  std::vector<std::string> addLegendTripleCouplings() const override;
  std::vector<std::string> addLegendVEV() const override;

  void set_gen(const std::vector<double> &par) override;
  void set_CT_Pot_Par(const std::vector<double> &par) override;
  void write() const override;

  void TripleHiggsCouplings() override;
  std::vector<double> calc_CT() const override;

  void SetCurvatureArrays() override;
  bool CalculateDebyeSimplified() override;
  bool CalculateDebyeGaugeSimplified() override;
  double VTreeSimplified(const std::vector<double> &v) const override;
  double VCounterSimplified(const std::vector<double> &v) const override;
  void Debugging(const std::vector<double> &input,
                 std::vector<double> &output) const override;
};

} // namespace Models
} // namespace BSMPT
