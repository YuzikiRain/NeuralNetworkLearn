using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MathNet.Numerics.LinearAlgebra.Double;

public class NeuralNetwork : MonoBehaviour
{
    public float Interval = 0.1f;
    public int Iteration = 100;
    public float LearnSpeed = 0.5f;
    public AnimationCurve DeviationCurve;
    [SerializeField] private float _time;

    private int layerCount;
    private List<NeuralLayer> layers = new List<NeuralLayer>();
    private float rate = 0.5f;
    DenseMatrix sample;
    DenseMatrix label;

    private void Update()
    {
        if (Input.GetKeyDown(KeyCode.Space)) { StartCoroutine(StartTrain()); }
    }

    private IEnumerator StartTrain()
    {
        Init();

        var iteration = 0;
        while (true)
        {
            for (int i = 0; i < Iteration; i++, iteration++)
            {
                // 正向传播
                Forward();

                // 反向传播
                Backward();

                // 根据梯度信息，更新权重和偏置
                UpdateWB(rate);

                //// 显示当前函数值与目标值
                //Show(deviation, iteration);
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
        int[] parameters = new int[] { 4, 4, 4 };
        layerCount = parameters.Length;
        sample = DenseMatrix.OfArray(new[,] { { 0.2, 0.1, 0.15, 0.1 }, }).Transpose() as DenseMatrix;
        label = DenseMatrix.OfArray(new[,] { { 0.4, 0.8, 0.2, 0.6 }, }).Transpose() as DenseMatrix;

        for (int i = 0; i < parameters.Length; i++)
        {
            layers.Add(new NeuralLayer(parameters[i], parameters[i], new SigmoidActivator()));
        }
        rate = 0.5f;
    }

    /// <summary>
    /// 正向传播
    /// </summary>
    private void Forward()
    {
        // output当作一个临时变量，仅用于迭代地进行前向传播
        var output = sample;
        for (int levelIndex = 0; levelIndex < layerCount; levelIndex++)
        {
            var layer = layers[levelIndex];
            output = layer.Forward(output);
        }
    }

    private void Backward()
    {
        // 1. 先计算输出层的后一层的误差
        var outputLayer = layers[layers.Count - 1];
        // -(t - y) y (1 - y)
        var delta = DenseMatrix.op_DotMultiply((label - outputLayer.output), outputLayer.activator.Backward(outputLayer.output)) as DenseMatrix;
        Debug.Log($"label = {label} outputLayer.output = {outputLayer.output}  deviation = {label - outputLayer.output} layer = {layers.Count - 1}");

        // 2. 再从输出层开始计算每一层的误差
        for (int i = layerCount - 1; i >= 0; i--)
        {
            var layer = layers[i];
            delta = layer.Backward(delta);
        }
        // 这里可以迭代到 1层，即输入层的后一层，第一个隐藏层
        // 迭代到 1 层时计算的delta虽然是错误的，但没关系因为这次返回的delta没被用到
        // 错误是因为上一层就是输入层了，其output=input，对其偏导应该为1，而不是sigmoid的output(1-output)
    }

    private void UpdateWB(float rate)
    {
        for (int i = 0; i < layers.Count; i++)
        {
            layers[i].UpdateWB(rate);
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

    private void Show(float deviation, int iteration)
    {
        DeviationCurve.AddKey(_time, deviation);
        Debug.Log($"deviation = {deviation} iteration = {iteration}");
    }
}
