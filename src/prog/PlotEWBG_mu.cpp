/*
 * PlotEWBG_mu.cpp
 *
 *
 *      Copyright (C) 2020  Philipp Basler, Margarete Mühlleitner and Jonas Müller

        This program is free software: you can redistribute it and/or modify
        it under the terms of the GNU General Public License as published by
        the Free Software Foundation, either version 3 of the License, or
        (at your option) any later version.

        This program is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        GNU General Public License for more details.

        You should have received a copy of the GNU General Public License
        along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
/**
 * @file
 * Calculates eta as a function of the renormalised scale mu. The renormalisation scale mu is varied from 1/2 to 1.5 C_vev0 in NumberOfStep steps.
 */

#include <bits/exception.h>                     // for exception
#include <stdlib.h>                             // for atoi, EXIT_FAILURE
#include <algorithm>                            // for copy, max
#include <memory>                               // for shared_ptr, __shared_...
#include <string>                               // for string, operator<<
#include <utility>                              // for pair
#include <vector>                               // for vector
#include <BSMPT/models/ClassPotentialOrigin.h>  // for Class_Potential_Origin
#include <BSMPT/models/IncludeAllModels.h>
#include <BSMPT/minimizer/Minimizer.h>
#include <BSMPT/baryo_calculation/transport_equations.h>
#include <BSMPT/baryo_calculation/CalculateEtaInterface.h>
#include <BSMPT/utility.h>
#include <iostream>
#include <iomanip>
#include <fstream>

using namespace std;
using namespace BSMPT;


auto getCLIArguments(int argc, char *argv[])
{
    struct ReturnType{
        BSMPT::ModelID::ModelIDs Model{};
        int Line{}, NumberOfSteps{};
        std::string InputFile, OutputFile,ConfigFile;
        bool TerminalOutput{false};
        double vw{0.1};
    };

    std::vector<std::string> args;
    for(int i{1};i<argc;++i) args.push_back(argv[i]);

    if(argc < 7 or args.at(0) == "--help")
    {
        int SizeOfFirstColumn = std::string("--TerminalOutput=           ").size();
        std::cout << "EWBGRenormScale calculates the strength of the EWBG while varying the MSBar renormalisation scale" << std::endl
                  << "It is called either by " << std::endl
                  << "./EWBGRenormScale Model Inputfile Outputfile  Line NumberOfSteps Configfile" << std::endl
                  << "or with the following arguments" << std::endl
                  << std::setw(SizeOfFirstColumn) << std::left<< "--help"
                  << "Shows this menu" << std::endl
                  << std::setw(SizeOfFirstColumn) << std::left << "--model="
                  << "The model you want to investigate"<<std::endl
                  << std::setw(SizeOfFirstColumn) << std::left<<"--input="
                  << "The input file in tsv format" << std::endl
                  << std::setw(SizeOfFirstColumn) << std::left<<"--output="
                  << "The output file in tsv format" << std::endl
                  << std::setw(SizeOfFirstColumn) << std::left<<"--Line="
                  <<"The line in the input file with the parameter point. Expects line 1 to be a legend." << std::endl
                  << std::setw(SizeOfFirstColumn) << std::left << "--config="
                  << "The EWBG config file." << std::endl
                  << std::setw(SizeOfFirstColumn) << std::left<<"--TerminalOutput="
                  <<"y/n Turns on additional information in the terminal during the calculation." << std::endl
                  << std::setw(SizeOfFirstColumn) << std::left << "--vw="
                  << "Wall velocity for the EWBG calculation. Default value of 0.1." << std::endl
                  << std::setw(SizeOfFirstColumn) << std::left << "--NumberOfSteps="
                  << "Number of Steps to vary the scale between 0.5 and 1.5 times the original scale." << std::endl;
        ShowInputError();
    }

    if(args.size() > 0 and args.at(0)=="--help")
    {
        throw int{0};
    }
    else if(argc < 7)
    {
        throw std::runtime_error("Too few arguments.");
    }


    ReturnType res;
    std::string prefix{"--"};
    bool UsePrefix = StringStartsWith(args.at(0),prefix);
    if(UsePrefix)
    {
        for(const auto& arg: args)
        {
            auto el = arg;
            std::transform(el.begin(), el.end(), el.begin(), ::tolower);
            if(StringStartsWith(el,"--model="))
            {
                res.Model = BSMPT::ModelID::getModel(el.substr(std::string("--model=").size()));
            }
            else if(StringStartsWith(el,"--input="))
            {
                res.InputFile = arg.substr(std::string("--input=").size());
            }
            else if(StringStartsWith(el,"--output="))
            {
                res.OutputFile = arg.substr(std::string("--output=").size());
            }
            else if(StringStartsWith(el,"--line="))
            {
                res.Line = std::stoi(el.substr(std::string("--line=").size()));
            }
            else if(StringStartsWith(el,"--numberofsteps="))
            {
                res.NumberOfSteps = std::stoi(el.substr(std::string("--numberofsteps=").size()));
            }
            else if(StringStartsWith(el,"--terminaloutput="))
            {
                res.TerminalOutput = el.substr(std::string("--lastline=").size()) == "y";
            }
            else if(StringStartsWith(el,"--vw="))
            {
                res.vw = std::stod(el.substr(std::string("--vw=").size()));
            }
            else if(StringStartsWith(el,"--config="))
            {
                res.ConfigFile = arg.substr(std::string("--config").size());
            }
        }
    }
    else{
        res.Model = ModelID::getModel(args.at(0));
        res.InputFile = args.at(1);
        res.OutputFile = args.at(2);
        res.Line = std::stoi(args.at(3));
        res.NumberOfSteps = std::stoi(args.at(4));
        res.ConfigFile = args.at(5);
        if(argc == 8) {
            std::string s7 = argv[6];
            res.TerminalOutput = ("y" == s7);
        }
    }

    if(res.NumberOfSteps == 0)
    {
        throw std::runtime_error("You have set the number of steps to zero.");
    }


    return res;
}

int main(int argc, char *argv[]) try{

    const auto args = getCLIArguments(argc,argv);

    if(args.Model==ModelID::ModelIDs::NotSet) {
        std::cerr << "Your Model parameter does not match with the implemented Models." << std::endl;
        ShowInputError();
        return EXIT_FAILURE;
    }

    //Init: Interface Class for the different transport methods
    Baryo::CalculateEtaInterface EtaInterface(argv[6] /* = Config file */);
    if(args.Line < 1)
    {
        std::cout << "Start line counting with 1" << std::endl;
        return EXIT_FAILURE;
    }
    int linecounter = 1;
    std::ifstream infile(args.InputFile);
    if(!infile.good()) {
        std::cout << "Input file not found " << std::endl;
        return EXIT_FAILURE;
    }
    std::ofstream outfile(args.OutputFile);
    if(!outfile.good())
    {
        std::cout << "Can not create file " << args.OutputFile << std::endl;
        return EXIT_FAILURE;
    }
    std::string linestr;
    std::shared_ptr<BSMPT::Class_Potential_Origin> modelPointer = ModelID::FChoose(args.Model);
    std::vector<std::string> etaLegend = EtaInterface.legend();// Declare the vector for the PTFinder algorithm
    std::size_t nPar,nParCT;
    nPar = modelPointer->get_nPar();
    nParCT = modelPointer->get_nParCT();
    std::vector<double> par(nPar);
    std::vector<double> parCT(nParCT);

    while(getline(infile,linestr))
    {
        if(linecounter > args.Line) break;

        if(linecounter == 1)
          {

            outfile << linestr<<sep << "mu_factor"<<sep<<"mu";
            for(auto x: modelPointer->addLegendTemp()) outfile << sep <<x+"_mu";
            outfile << sep << "BSMPT_StatusFlag";
            outfile << sep << "vw";
            outfile << sep << "L_W";
            outfile << sep << "top_sym_phase";
            outfile << sep << "top_brk_phase";
            outfile << sep << "bot_sym_phase";
            outfile << sep << "bot_brk_phase";
            outfile << sep << "tau_sym_phase";
            outfile << sep << "tau_brk_phase";
            for(auto x: etaLegend) outfile << sep << x+"_muvar";
            outfile << std::endl;

            modelPointer->setUseIndexCol(linestr);
          }
        if(args.Line==linecounter)
        {
            std::pair<std::vector<double>,std::vector<double>> parameters = modelPointer->initModel(linestr);
            par=parameters.first;
            parCT = parameters.second;
            if(args.TerminalOutput) modelPointer->write();
            if(args.TerminalOutput)
            {
                std::cout<<"Calculating EWPT in default settings with:\n mu = "<<modelPointer->get_scale()
                        <<std::endl;
            }
            if(args.TerminalOutput)std::cout<<"Start of mu variation"<<std::endl;
            for(int step=0;step<args.NumberOfSteps;step++){
                double mu_factor = 1/2. + (step/static_cast<double>(args.NumberOfSteps));
                if(args.TerminalOutput) std::cout<<"\r currently mu_factor = "<<mu_factor<<std::endl;
                auto VEVnames = modelPointer->addLegendTemp();
                auto CT_mu=modelPointer->resetScale(C_vev0*mu_factor);
                auto EWPT_mu = Minimizer::PTFinder_gen_all(modelPointer,0,300);
                std::vector<double> startpoint;
                for(const auto& x : EWPT_mu.EWMinimum) startpoint.push_back(x/2.);
                if(EWPT_mu.StatusFlag==1){
                    std::vector<double>checkmu;
                    auto VEV_mu_sym = Minimizer::Minimize_gen_all(modelPointer,EWPT_mu.Tc+1,checkmu,startpoint);
                    auto VEV_mu_brk = EWPT_mu.EWMinimum;
                    auto eta_mu = EtaInterface.CalcEta(args.vw,EWPT_mu.EWMinimum,VEV_mu_sym,EWPT_mu.Tc,modelPointer);

                    outfile << linestr;
                    outfile << sep << mu_factor <<sep<< mu_factor*C_vev0;
                    outfile << sep << EWPT_mu.Tc<<sep<<EWPT_mu.vc<<sep<<EWPT_mu.vc/EWPT_mu.Tc<<sep<<EWPT_mu.EWMinimum;
                    outfile << sep << EWPT_mu.StatusFlag;
                    outfile << sep << args.vw;
                    outfile << sep << EtaInterface.getLW();
                    outfile << sep << EtaInterface.GSL_integration_mubl_container.getSymmetricCPViolatingPhase_top();
                    outfile << sep << EtaInterface.GSL_integration_mubl_container.getBrokenCPViolatingPhase_top();
                    outfile << sep << EtaInterface.GSL_integration_mubl_container.getSymmetricCPViolatingPhase_bot();
                    outfile << sep << EtaInterface.GSL_integration_mubl_container.getBrokenCPViolatingPhase_bot();
                    outfile << sep << EtaInterface.GSL_integration_mubl_container.getSymmetricCPViolatingPhase_tau();
                    outfile << sep << EtaInterface.GSL_integration_mubl_container.getBrokenCPViolatingPhase_tau();
                    for(auto x: eta_mu) outfile << sep << x;

                    outfile << std::endl;


                }//EWPT@mu found
                else {
                    if(args.TerminalOutput) std::cout<<"\tNo SFOEWPT found for given scale"<<std::endl;
                    continue;
                }
        }//END: Mu Factor
        }//END:LineCounter
        linecounter++;
        if(infile.eof()) break;
    }
    if(args.TerminalOutput) std::cout << std::endl;
    outfile.close();

    return EXIT_SUCCESS;

}
catch(int)
{
    return EXIT_SUCCESS;
}
catch(exception& e){
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
}
