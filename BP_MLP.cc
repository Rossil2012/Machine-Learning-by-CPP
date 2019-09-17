#include "BP_MLP.hh"
using namespace AI_Library;

//to be deleted
double random_ini()
{
    std::default_random_engine e;
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    e.seed(time(0));
    return u(e);
}

//Activate Functions
///sigmoid
double sigmoid(double x)
{
    return (1 / (1 + exp(-x)));
}
///tanh
double tanh(double x)
{
    return ((exp(x) - exp(-x)) / (exp(x) + exp(-x)));
}
///ReLU
double ReLU(double x)
{
    return ((x <= 0)?(0.1 * x):x);
}

//The Derived Functions of Activate Functions
///d_sigmoid
double d_sigmoid(double x)
{
    return (sigmoid(x) * (1 - sigmoid(x)));
}
///d_tanh
double d_tanh(double x)
{
    return (1 - tanh(x) * tanh(x));
}
///d_ReLU
double d_ReLU(double x)
{
    return ((x <= 0)?0.1:1);
}

//Learning Rate Modes //to be revised
//double manual()

BP_MLP_Layer::BP_MLP_Layer(int len)
    : Neuron(new double[len]), Error(new double[len]), Length(len)
{
}

BP_MLP_Layer::~BP_MLP_Layer()
{
    delete []Neuron;
    delete []Error;
}

void BP_MLP_Layer::Set_Layer(double *input, int length)
{
    if (length != Length)
    {
        assert("Wrong Length of Input Layer");
    }
    else 
    {
        for (int i = 0; i < length; ++i)
        {
            Neuron[i] = input[i];
        }
    }
}

double &BP_MLP_Layer::rN(int idx)
{
    if (idx >= Length)
    {
        assert("Out of Range!");
    }
    return Neuron[idx];
}

double &BP_MLP_Layer::rE(int idx)
{
    if (idx >= Length)
    {
        assert("Out of Range!");
    }
    return Error[idx];
}


BP_MLP_Net::BP_MLP_Net(int in_num, int hid_num, int out_num, int hid_layer = 3)
    : In_Neuron_Num(in_num), Hid_Layer_Num(hid_layer), Hid_Neuron_Num(hid_num), Out_Neuron_Num(out_num),
      Input_Layer(new BP_MLP_Layer(in_num)), Output_Layer(new BP_MLP_Layer(out_num))
{
    //Create layers
    Hidden_Layer = new BP_MLP_Layer *[hid_layer];
    for (int i = 0; i < hid_layer; ++i)
    {
        Hidden_Layer[i] = new BP_MLP_Layer(hid_num);
    }

    //Initialize Weights
    std::default_random_engine e;
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    e.seed(time(0));

    ///in_to_hid_Weights
    in_to_hid_Weights = new double *[hid_num];
    for (int i = 0; i < hid_num; ++i)
    {
        in_to_hid_Weights[i] = new double[in_num];
    }

    ///hid_to_hid_Weights
    hid_to_hid_Weights = new double **[hid_layer - 1];
    for (int i = 0; i < hid_layer - 1; ++i)
    {
        hid_to_hid_Weights[i] = new double *[hid_num];
    }

    for (int i = 0; i < hid_layer - 1; ++i)
        for(int j = 0; j < hid_num; ++j)
        {
            hid_to_hid_Weights[i][j] = new double[hid_num];
        }
    
    ///hid_to_out_Weights
    hid_to_out_Weights = new double *[out_num];
    for (int i = 0; i < out_num; ++i)
    {
        hid_to_out_Weights[i] = new double[hid_num];
    }

    //Initialize Bias
    ///in_to_hid_Bias
    in_to_hid_Bias = new double[hid_num];

    ///hid_to_hid_Bias
    hid_to_hid_Bias = new double *[hid_layer - 1];
    for (int i = 0; i < hid_layer - 1; ++i)
    {
        hid_to_hid_Bias[i] = new double[hid_num];
    }

    ///hid_to_out_Bias
    hid_to_out_Bias = new double[out_num];

    //Set Activate Function  (default == SIGMOID)
    Set_Activate_Function(SIGMOID);
    


}

BP_MLP_Net::~BP_MLP_Net()
{
    //Layers
    delete Input_Layer;
    delete Output_Layer;
    for (int i = 0; i < Hid_Layer_Num; ++i)
    {
        delete Hidden_Layer[i];
    }
    delete []Hidden_Layer;

    //Expectation
    delete []Expect_Output;

    //Bias
    delete []in_to_hid_Bias;
    delete []hid_to_out_Bias;
    for (int i = 0; i < Hid_Layer_Num - 1; ++i)
    {
        delete []hid_to_hid_Bias[i];
    }
    delete []hid_to_hid_Bias;

    //Weights
    for (int i = 0; i < Hid_Neuron_Num; ++i)
    {
        delete []in_to_hid_Weights[i];
    }
    delete []in_to_hid_Weights;
    for (int i = 0; i < Out_Neuron_Num; ++i)
    {
        delete []hid_to_out_Weights[i];
    }
    delete []hid_to_out_Weights;
    for (int i = 0; i < Hid_Layer_Num - 1; ++i)
    {
        for (int j = 0; j < Hid_Neuron_Num; ++j)
        {
            delete []hid_to_hid_Weights[i][j];
        }
        delete []hid_to_hid_Weights[i];
    }
    delete []hid_to_hid_Weights;
        
}

void BP_MLP_Net::Set_Activate_Function(Activate_Function mode)
{
    Act_Mode = mode;

    //Set_Activate_Function
    if (Act_Mode == SIGMOID)
    {
        Act_Func = &sigmoid;
        d_Act_Func = &d_sigmoid;
        Initialize(-1.0, 1.0);
        L_Rate = 2.0;
    }
    else if (Act_Mode == TANH)
    {
        Act_Func = &tanh;
        d_Act_Func = &d_tanh;
        Initialize(-1.0, 1.0);
        L_Rate = 0.2;
    }
    else if (Act_Mode == RELU)
    {
        Act_Func = &ReLU;
        d_Act_Func = &d_ReLU;
        Initialize(-0.1, 0.1);
        L_Rate = 0.2;
    }


}

void BP_MLP_Net::Set_Learning_Rate(Learning_Rate mode)
{
    L_Mode = mode;
}

void BP_MLP_Net::Set_Franchise(double franchise)
{
    Franchise = franchise;
}

void BP_MLP_Net::feed(double *input, int in_length, double *output, int out_length)
{
    if (in_length != In_Neuron_Num)
    {
        assert("Length of Input is Out of Range");
    }
    if (out_length != Out_Neuron_Num)
    {
        assert("Length of Output is Out of Range");
    }

    Input_Layer->Set_Layer(input, in_length);

    if (Expect_Output)
    {
        delete Expect_Output;
    }
    Expect_Output = new double[out_length];

    for (int i = 0; i < out_length; ++i)
    {
        Expect_Output[i] = output[i];
    }
}

void BP_MLP_Net::Initialize(double floor, double top)
{
    std::default_random_engine e;
    std::uniform_real_distribution<double> u(floor, top);
    e.seed(time(0));

    //in_to_hid_Weights
    for (int i = 0; i < Hid_Neuron_Num; ++i)
        for (int j = 0; j < In_Neuron_Num; ++j)
        {
            in_to_hid_Weights[i][j] = u(e);
        }

    //hid_to_hid_Weights
    for (int i = 0; i < Hid_Layer_Num - 1; ++i)
        for (int j = 0; j < Hid_Neuron_Num; ++j)
            for(int k = 0; k < Hid_Neuron_Num; ++k)
            {
                hid_to_hid_Weights[i][j][k] = u(e);
            }
    
    //hid_to_out_Weights
    for (int i = 0; i < Out_Neuron_Num; ++i)
        for (int j = 0; j < Hid_Neuron_Num; ++j)
        {
            in_to_hid_Weights[i][j] = u(e);
        }

    //Bias
    ///in_to_hid_Bias
    for (int i = 0; i < Hid_Neuron_Num; ++i)
    {
        in_to_hid_Bias[i] = u(e);
    }
    ///hid_to_hid_Bias
    for (int i = 0; i < Hid_Layer_Num - 1; ++i)
        for (int j = 0 ; j < Hid_Neuron_Num; ++j)
        {
            hid_to_hid_Bias[i][j] = u(e);
        }
    ///in_to_out_Bias
    for (int i = 0; i < Out_Neuron_Num; ++i)
    {
        hid_to_out_Bias[i] = u(e);
    }
}

void BP_MLP_Net::Foreward_Propagate()
{
    //Update Hidden_Layer[0]
    for (int i = 0; i < Hid_Neuron_Num; ++i)
    {
        double tmp = 0;
        for (int j = 0; j < In_Neuron_Num; ++j)
        {
            tmp += Act_Func(Input_Layer->rN(j)) * in_to_hid_Weights[i][j];
        }
        Hidden_Layer[0]->rN(i) = tmp + in_to_hid_Bias[i];
    }

    //Update the Rest Hidden_Layer
    for (int i = 1; i < Hid_Layer_Num; ++i)
        for (int j = 0; j < Hid_Neuron_Num; ++j)
        {
            double tmp = 0;
            for (int k = 0; k < Hid_Neuron_Num; ++k)
            {
                tmp += Act_Func(Hidden_Layer[i - 1]->rN(k)) * hid_to_hid_Weights[i - 1][j][k];
            }
            Hidden_Layer[i]->rN(j) = tmp + hid_to_hid_Bias[i - 1][j];
        }

    //Update the Output_Layer
    for (int i = 0; i < Out_Neuron_Num; ++i)
    {
        double tmp = 0;
        for (int j = 0; j < Hid_Neuron_Num; ++j)
        {
            tmp += Act_Func(Hidden_Layer[Hid_Layer_Num - 1]->rN(j)) * hid_to_out_Weights[i][j];
        }
        Output_Layer->rN(i) = tmp + hid_to_out_Bias[i];
        //In case of nan
        if (isnan(Output_Layer->rN(i)))
        {
            if (Act_Mode == SIGMOID)
            {
                Initialize(-1.0, 1.0);
            }
            else if (Act_Mode == TANH)
            {
                Initialize(-1.0, 1.0);
            }
            else if (Act_Mode == RELU)
            {
                Initialize(-0.1, 0.1);
            }       
        }
    }

    //Accumulate the Error of the Output Layer
    Sum_Error = 0;
    for (int i = 0; i < Out_Neuron_Num; ++i)
    {
        Sum_Error += fabs(Act_Func(Output_Layer->rN(i)) - Act_Func(Expect_Output[i]));
    }
}

void BP_MLP_Net::Back_Propagate()
{
    //Set the Error of the Output Layer
    for (int i = 0; i < Out_Neuron_Num; ++i)
    {
        Output_Layer->rE(i) = (Act_Func(Output_Layer->rN(i)) - Act_Func(Expect_Output[i])) * d_Act_Func(Output_Layer->rN(i));
    }

    //Adjust hid_to_out_Weights && hid_to_out_Bias
    for (int i = 0; i < Out_Neuron_Num; ++i)
    {
        for (int j = 0; j < Hid_Neuron_Num; ++j)
        {
            hid_to_out_Weights[i][j] -= L_Rate * Output_Layer->rE(i) * Act_Func(Hidden_Layer[Hid_Layer_Num - 1]->rN(j));
        }

        hid_to_out_Bias[i] -= L_Rate * Output_Layer->rE(i);
    }

    //Adjust hid_to_hid_Weights && hid_to_hid_Bias
    for (int i = Hid_Layer_Num - 2; i >= 0; --i)
        for (int j = 0; j < Hid_Neuron_Num; ++j)
        {
            for (int k = 0; k < Hid_Neuron_Num; ++k)
            {
                hid_to_hid_Weights[i][j][k] -= L_Rate * Hidden_Layer[i + 1]->rE(j) * Act_Func(Hidden_Layer[i]->rN(k));
            }

            hid_to_hid_Bias[i][j] -= L_Rate * Hidden_Layer[i + 1]->rE(j);
        }
    
    //Adjust in_to_hid_Weights && in_to_hid_Bias
    for (int i = 0; i < Hid_Neuron_Num; ++i)
    {
        for (int j = 0; j < In_Neuron_Num; ++j)
        {
            in_to_hid_Weights[i][j] -= L_Rate * Hidden_Layer[0]->rE(i) * Act_Func(Input_Layer->rN(j));
        }

        in_to_hid_Bias[i] -= L_Rate * Hidden_Layer[0]->rE(i);
    }

}

void BP_MLP_Net::run()
{
    Foreward_Propagate();
    while(Sum_Error > Franchise)
    {
        Back_Propagate();
        Foreward_Propagate();
    }
}

void BP_MLP_Net::get_output(double *container, int length)
{
    if (length != Out_Neuron_Num)
    {
        assert("Wrong Length of Container!");
    }

    for (int i = 0; i < length; ++i)
    {
        container[i] = Output_Layer->rN(i);
    }
}
