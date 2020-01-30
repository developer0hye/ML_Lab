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

class LogisticRegression {
public:

    LogisticRegression() :m(0.1), c(0.1), lr(0.0001) {}

    double predict(double x) //Y= mX+c
    {
        double gx = m*x + c;
        double fx = 1./(exp(-gx)+1.);
        return fx;
    }

    void set_learning_rate(double lr)
    {
        this->lr = lr;
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
            double error = this->predict(x[i]) - target_y[i];
            dm += error*x[i];
            dc += error;
        }
        dm /= n;
        dc /= n;

        m = m - lr * dm;
        c = c - lr * dc;
    }

    double compute_cost(vector<double> x, vector<double> target_y)
    {
        assert(x.size() != 0);
        assert(x.size() == target_y.size());

        double sum_cost = 0.0;
        for (int i = 0; i < x.size(); i++)
        {
            double fx = this->predict(x[i]);
            double cost = target_y[i]*log(fx+1e-6) + (1-target_y[i])*log( (1.-fx)+1e-6);
            sum_cost += -cost;
        }
        return sum_cost / (double)(x.size());
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
    LogisticRegression model;

    vector<double> train_x, test_x; //x[i]: xi
    vector<double> train_target_y, test_target_y; //y[i]: yi

    load_dataset("train.txt", train_x, train_target_y);
    load_dataset("test.txt", test_x, test_target_y);

    double lr = 1e-2;
    model.set_learning_rate(lr);
    //train
    for (int i = 0; i < 1000000; i++)
    {
        model.gradient_descent(train_x, train_target_y);
        if (i % 100000 == 0)
        {
            cout << "train error: " << model.compute_cost(train_x, train_target_y) << endl;
            cout << "test error: " << model.compute_cost(test_x, test_target_y) << endl;
            cout <<  "lr: " << lr << endl;
            lr *= 0.5;
            model.set_learning_rate(lr);
        }
    }

    model.print_model_params();

    int nAnswer = 0;
    for (int i = 0; i < test_x.size(); i++)
    {
        double predicted_grade = model.predict(test_x[i]) >= 0.5? 1.0: 0.0;
        if (predicted_grade == test_target_y[i]) nAnswer++;
    }

    cout << "answer: " << nAnswer << endl;
    cout << "accuracy: " << nAnswer*100./(double)test_x.size() << endl;

    return 0;
}
