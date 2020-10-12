using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MathNet.Numerics.LinearAlgebra.Double;

public class DeepLearn : MonoBehaviour
{
    public float Interval = 0.1f;
    public int Iteration = 100;
    public float LearnSpeed = 0.5f;
    public AnimationCurve DeviationCurve;
    [SerializeField] private float _time;

    int level;
    Dictionary<string, DenseMatrix> parameterMatrix = new Dictionary<string, DenseMatrix>();
    DenseMatrix inputMatrix;
    Dictionary<string, DenseMatrix> cacheOutputMatrix = new Dictionary<string, DenseMatrix>();

    private void Update()
    {
        if (Input.GetKeyDown(KeyCode.Space)) { StartCoroutine(StartLearn()); }
    }

    private IEnumerator StartLearn()
    {


        // 初始化所有参数
        //var LearnSpeed = 0.5f;
        // 第1层为输入层，2为隐藏层，3为输出层
        var w11to21 = 1f;
        var w11to22 = 1f;
        var w12to21 = 1f;
        var w12to22 = 1f;
        var w21to31 = 1f;
        var w22to31 = 1f;

        var b21 = 1f;
        var b22 = 1f;
        var b31 = 1f;

        float[] x1Array = new float[] { 0.1f, 0.2f, 0.3f, 0.4f };
        float[] x2Array = new float[] { 0.1f, 0.2f, 0.3f, 0.4f };
        float[] tArray = new float[] { 0.5f, 0.6f, 0.7f, 0.8f };

        var iteration = 0;
        while (true)
        {
            for (int i = 0; i < Iteration; i++, iteration++)
            {
                var t = tArray[i];
                var x1 = x1Array[i];
                var x2 = x2Array[i];
                // 计算输出值
                var h1 = x1 * w11to21 + x2 * w12to21 + b21;
                var h2 = x1 * w11to22 + x2 * w12to22 + b22;
                var o1 = h1 * w21to31 + h2 * w22to31 + b31;
                var y = Sigmoid(o1);

                var deviation = Mathf.Abs(y - t);
                // 训练，即更新权重和偏置，让神经网络误差更小
                w11to21 = w11to21 - LearnSpeed * deviation * W11to21Derivative(y, t, w21to31, x1);
                w11to22 = w11to22 - LearnSpeed * deviation * W11to22Derivative(y, t, w22to31, x1);
                w12to21 = w12to21 - LearnSpeed * deviation * W12to21Derivative(y, t, w21to31, x1);
                w12to22 = w12to22 - LearnSpeed * deviation * W12to22Derivative(y, t, w22to31, x1);

                b31 = b31 - LearnSpeed * B31Derivative(y, t);
                b21 = b21 - LearnSpeed * B21Derivative(y, t, w21to31);
                b22 = b22 - LearnSpeed * B22Derivative(y, t, w22to31);

                w21to31 = w21to31 - LearnSpeed * W21to31Derivative(y, t, h1);
                w22to31 = w22to31 - LearnSpeed * W22to31Derivative(y, t, h2);

                // 显示当前函数值与目标值
                Show(deviation, iteration);
                yield return new WaitForSeconds(Interval);
                _time += Time.deltaTime;
            }

        }
    }

    /// <summary>
    /// 初始化所有参数
    /// </summary>
    private void Init()
    {
        int[] parameters = new int[] { 4, 4, 4, 4 };
        inputMatrix = new DenseMatrix(parameters[0], 1);
        level = parameters.Length;
        for (int i = 0; i < level; i++)
        {
            // 以第0层即输入层的神经元数作为统一的每层神经元数，除去输出层需要计算level-1层
            parameterMatrix[$"w{i}"] = new DenseMatrix(parameters[0], level - 1);
        }
    }

    /// <summary>
    /// 正向传播
    /// </summary>
    private void Forward(DenseMatrix x)
    {
        DenseMatrix output = DenseMatrix.OfMatrix(inputMatrix);
        DenseMatrix input = null;

        for (int levelIndex = 0; levelIndex < level; levelIndex++)
        {
            // 通过上一层的输出得到这一层的输入
            input = output;
            output = parameterMatrix[$"w{levelIndex}"] * input;
            //cacheOutputMatrix[$""] = output;
        }
    }

    private float Sigmoid(float x)
    {
        return 1f / (Mathf.Exp(-x) + 1);
    }

    private float SigmoidDerivative(float o1)
    {
        var s = Sigmoid(o1);
        return s * (1 - s);
    }

    private float W21to31Derivative(float y, float t, float h1)
    {
        return -1 * (t - y) * y * (1 - y) * h1;
    }

    private float W22to31Derivative(float y, float t, float h2)
    {
        return -1 * (t - y) * y * (1 - y) * h2;
    }

    private float W11to21Derivative(float y, float t, float w21to31, float x1)
    {
        return -1 * (t - y) * y * (1 - y) * w21to31 * x1;
    }

    private float W11to22Derivative(float y, float t, float w22to31, float x1)
    {
        return -1 * (t - y) * y * (1 - y) * w22to31 * x1;
    }

    private float W12to21Derivative(float y, float t, float w21to31, float x2)
    {
        return -1 * (t - y) * y * (1 - y) * w21to31 * x2;
    }

    private float W12to22Derivative(float y, float t, float w22to31, float x2)
    {
        return -1 * (t - y) * y * (1 - y) * w22to31 * x2;
    }

    private float B31Derivative(float y, float t)
    {
        return -1 * (t - y) * y * (1 - y);
    }

    private float B21Derivative(float y, float t, float w21to31)
    {
        return B31Derivative(y, t) * w21to31;
    }

    private float B22Derivative(float y, float t, float w22to31)
    {
        return B31Derivative(y, t) * w22to31;
    }

    private void Show(float deviation, int iteration)
    {
        DeviationCurve.AddKey(_time, deviation);
        Debug.Log($"deviation = {deviation} iteration = {iteration}");
    }
}
