#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <string>
#include <cassert>
#include <cmath>

using namespace std;

void load_dataset(string path, vector<double>& x, vector<double>& target_y)
{
    assert(x.empty() && target_y.empty());

    ifstream inFile(path);
    string in_line;

    while (getline(inFile, in_line)) {
        std::istringstream iss(in_line);

        std::string x_str, target_y_str;

        std::getline(iss, x_str, '\t');
        std::getline(iss, target_y_str, '\t');

        x.push_back(stod(x_str));
        target_y.push_back(stod(target_y_str));
    }

    inFile.close();
}

class LinearRegression {
public:

    LinearRegression() :m(0.1), c(0.1), lr(0.000165) {}

    double predict(double x) //Y= mX+c
    {
        return x * m + c;
    }


    void gradient_descent(vector<double>& x, vector<double>& target_y)
    {
        assert(x.size() != 0);
        assert(x.size() == target_y.size());

        double dm = 0.0; //cost 를 m 로 편미분한 값
        double dc = 0.0; //cost 를 c 로 편미분한 값
        double n = x.size();

        for (int i = 0; i < x.size(); i++)
        {
            //여기다 코드 작성
            dm += 2 * x[i] * (target_y[i] - predict(x[i]));
            dc += 2 * (target_y[i] - predict(x[i]));



        }

        dm /= -n; //2 안곱함
        dc /= -n; //2 안곱함

        m = m - lr * dm;
        c = c - lr * dc;
    }

    double compute_mean_abs_error(vector<double> x, vector<double> target_y)
    {
        assert(x.size() != 0);
        assert(x.size() == target_y.size());

        double sum_error = 0.0;
        for (int i = 0; i < x.size(); i++) sum_error += abs(predict(x[i]) - target_y[i]);

        return sum_error / (double)(x.size());
    }

    void print_model_params()
    {
        cout << "m: " << m << ", c: " << c << endl;
    }

private:
    double m;
    double c;
    double lr;
};

int main()
{
    LinearRegression model;

    vector<double> train_x, test_x; //x[i]: xi
    vector<double> train_target_y, test_target_y; //y[i]: yi

    load_dataset("train.txt", train_x, train_target_y);
    load_dataset("test.txt", test_x, test_target_y);

    //train
    for (int i = 0; i < 10000; i++)
    {
        model.gradient_descent(train_x, train_target_y);
        if (i % 1000 == 0)
        {
            cout << "error: " << model.compute_mean_abs_error(train_x, train_target_y) << endl;
        }
    }

    model.print_model_params();

    int nAnswer = 0;
    for (int i = 0; i < test_x.size(); i++)
    {
        if (abs(model.predict(test_x[i]) - test_target_y[i]) <= 10) nAnswer++;
    }

    cout << "answer: " << nAnswer << endl;

    return 0;
}