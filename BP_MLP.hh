#pragma once
#include <random>
#include <cmath>
//#include <Vector>

//Threshold???????????????

namespace AI_Library
{

enum Activate_Function{SIGMOID, TANH, RELU};
enum Learning_Rate{MANUAL};  // to be revised

class BP_MLP_Layer
{
private:
    double *Neuron;
    double *Error;
    int Length;
public:
    BP_MLP_Layer(int);
    ~BP_MLP_Layer();
    void Set_Layer(double *, int);
    double &rN(int);
    double &rE(int);
};

class BP_MLP_Net
{
private:
    //Input Layer
    BP_MLP_Layer *Input_Layer;
    int In_Neuron_Num;

    //Hidden Layer
    BP_MLP_Layer **Hidden_Layer;
    int Hid_Layer_Num;                                              // Default of hidden layers' number is 3
    int Hid_Neuron_Num;

    //Output Layer
    BP_MLP_Layer *Output_Layer;
    int Out_Neuron_Num;

    //Expect Output
    double *Expect_Output = nullptr;

    //Learning Rate
    double L_Rate = 0.01;

    //Allowance of the Sum of Errors                                // Default of Franchise is 0.01
    double Franchise = 0.01;     

    //
    double Sum_Error;                                   

    double **in_to_hid_Weights;
    double ***hid_to_hid_Weights;
    double **hid_to_out_Weights;
    double *in_to_hid_Bias;
    double **hid_to_hid_Bias;
    double *hid_to_out_Bias;

    Activate_Function Act_Mode = SIGMOID;                           // Default of activate function is sigmoid
    Learning_Rate L_Mode = MANUAL;                                  // Default of learning rate mode is manual

    double (*Act_Func)(double);
    double (*d_Act_Func)(double);
    double (*cnt_Act_Func)(double);


    void Initialize(double, double);
    void Foreward_Propagate();
    void Back_Propagate();


public:


    BP_MLP_Net(int, int, int, int);
    ~BP_MLP_Net();

    void Set_Activate_Function(Activate_Function);
    void Set_Learning_Rate(Learning_Rate);
    void Set_Franchise(double);

    void initialize();
    void feed(double *, int, double *, int);
    void run();
    void get_output(double *, int);
    
};







};