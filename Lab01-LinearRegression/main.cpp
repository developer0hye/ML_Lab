#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cassert>
#include <cmath>

using namespace std;

class LinearRegression{
public:

    LinearRegression():m(0.1),c(0.1),lr(0.001){}

    double predict(double x) //Y= mX+c
    {
        return x*m+c;
    }


    void gradient_descent(vector<double> x, vector<double> target_y)
    {
        assert(x.size()!=0);
        assert(x.size()==target_y.size());

        double dm = 0.0;
        double dc = 0.0;
        double n = x.size();

        for(int i = 0; i < x.size(); i++)
        {
            double error = target_y[i]-predict(x[i]);

            dm += x[i]*error;
            dc += error;
        }

        dm = dm*(-2./n);
        dc = dc*(-2./n);

        m = m - lr*dm;
        c = c - lr*dc;
    }

    double compute_mean_error(vector<double> x, vector<double> target_y)
    {
        assert(x.size()!=0);
        assert(x.size()==target_y.size());

        double sum_error = 0.0;
        for(int i = 0; i < x.size(); i++) sum_error += abs(predict(x[i])-target_y[i]);

        return sum_error/(double)(x.size());
    }

    void print_model_params()
    {
        cout << "m: "<< m <<", " << c << endl;
    }

private:
    double m;
    double c;
    double lr;
};

int main()
{
    LinearRegression model;

    vector<double> x = {1.0, 2.0, 3.0}; //x[i]: xi
    vector<double> target_y = {2.5, 4.5, 6.5}; //y[i]: yi

    //train
    for(int i = 0; i < 100000; i++)
    {
        model.gradient_descent(x, target_y);
        if(i%1000 == 0) 
        {
            cout<< "error: " << model.compute_mean_error(x, target_y) << endl;
            model.print_model_params();
        }
    }
    
    model.print_model_params();

    for(int i = 0; i < x.size(); i++)
    {
        cout <<"x: " << x[i] << ", predicted_y: " << model.predict(x[i]) << ", "<< "target_y: " << target_y[i] << endl;
    }

    return 0;
}
