#include <iostream>
#include <ctime>
#define sigmoid(x) ( 1.0/(1.0+exp(-(x))) ) // sigmoid

using namespace std;

class Neuron {

    int input_num; //��ǲ
    double output; //�ƿ�ǲ
    double Alpha; //���� = �ΰ������
    double* input_weight; //��ǲ�� ���� ����ġ(�ó��� ���ᰭ��) 
    double* weight_error; //����ġ ����
    //����ġ�� ����ġ ������ �ϳ��Ǽ��� ������ ������ ����� �𸣱� ������ �迭��� �����ͷ� ����
public:
    Neuron(int input, double alpha) //������ �Լ�
    {
        input_num = input;
        Alpha = alpha;
        input_weight = new double[input + 1]; //��ġ���� ����� ��� ����ġ(w3) �Է� ��� => +1
        weight_error = new double[input + 1]; //��ġ���� ����� ��� ����ġ(w3) �Է� ��� => +1

        for (int i = 0; i < input + 1; i++)
        {
            input_weight[i] = ((double)rand() / RAND_MAX) - 1; //0.0 ~ 1.0������ �� -1 => ������ ����ġ(-1~1)�� ����
            weight_error[i] = 0.0; //����ġ���� �ʱ�ȭ
        }

    }
    ~Neuron()
    {
        delete[] input_weight;
        delete[] weight_error;
    }

    double work(double input[]) //�Է½�ȣ �迭�� ���ڷ� ����, �ñ׸��̵� �Լ��� �̿��Ͽ� �� ���
    {
        double sum = 0;
        for (int i = 0; i < input_num; i++)
        {
            sum += input[i] * input_weight[i]; //sum += ��ǲ X ����ġ
        }
        sum += input_weight[input_num] * 1.0; //sum +=�������ġ,����

        return sigmoid(sum);
    }


    void learn(double input[], double target) //�н��Լ�, neuron->learn(sample_input[j], sample_output[j]); �̷��� ȣ���ϱ� ������ input�� �迭��, target �ϳ��� ������ �޾Ƶ���
    {
        output = work(input); //���
        double output_error = output - target; //���� = ���-��ǥ,
        for (int i = 0; i < input_num; i++)
        {
            weight_error[i] += output_error * input[i]; //����ġ ���
        }
        weight_error[input_num] += output_error * 1.0;//�������ġ,����

    }

    void fix() //����ġ �����Լ�
    {
        for (int i = 0; i < input_num + 1; i++)
        {
            input_weight[i] -= Alpha * weight_error[i] * output * (1 - output);; //����(=�ΰ���, �н���(learning rate)����), output * (1 - output)->����, ���Ŀ� ���� ���� ����ġ ����
            weight_error[i] = 0.0;
        }
    }
};


int main() {
    srand((unsigned)time(NULL));	// ���� ����ġ�� ���Ƿ� ���ϱ� ���� ����.

    // ���� Ŭ���� ������.
    // Neuron(int num_of_input, double alpha) //���Ĵ� �ΰ��� ���
                  //(�Է��� ��, learning rate)
   
    Neuron OR(2, 0.1);
    Neuron NOT(1,0.1);
    Neuron AND(2, 0.1);
    Neuron NAND(2,0.1);
   
    double NOT_input[2][1] = {0, 1}; //��ǲ�� ���� 
    double NOT_output[2] = { 1, 0 };
    double sample_input[4][2] = { {0,0},{0,1},{1,0},{1,1} };
    double OR_output[4] = { 0, 1, 1, 1 };
    double AND_output[4] = { 0, 0, 0, 1 };
    double NAND_output[4] = { 1, 1, 1, 0 };

    for (int i = 0; i < 5000; i++)
    {
        for (int j = 0; j <2; j++)  
        {
            NOT.learn(NOT_input[j], NOT_output[j]);
        }
        NOT.fix();
        for (int j = 0; j < 4; j++) 
        {
            OR.learn(sample_input[j], OR_output[j]);
            AND.learn(sample_input[j], AND_output[j]);
            NAND.learn(sample_input[j], NAND_output[j]);
        }
        OR.fix();
        AND.fix();
        NAND.fix();

        // Print result //
        if ((i + 1) % 100 == 0)
        {
            cout << "------ Learn " << i + 1 << " times -----" << endl;
            for (int j = 0; j < 4; j++) {
                cout << sample_input[j][0] << ' ' << sample_input[j][1] << " : ";
                double a[2]{ OR.work(sample_input[j]), NAND.work(sample_input[j]) };
                cout << AND.work(a) <<endl;
            }
        }
    }
    return 0;
}

    
 
