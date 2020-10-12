using MathNet.Numerics.LinearAlgebra.Double;

public class NeuralLayer
{
    public Activator activator;
    private DenseMatrix input, w, b, wGradient, bGradient;
    public DenseMatrix output;

    /// <summary>
    /// 
    /// </summary>
    /// <param name="inputSize">y = wx + b，该参数表示w向量的分量个数</param>
    /// <param name="outputSize">每一层神经元的数量</param>
    /// <param name="activator">激活函数</param>
    public NeuralLayer(int inputSize, int outputSize, Activator activator)
    {
        this.activator = activator;
        // 初始化 row 为inputSize，column为outputSize，w_ij表示第L-1层的第 i 个神经元到第L层的第 j 个神经元的w
        var distribution = new MathNet.Numerics.Distributions.Normal(0, 1) { RandomSource = new System.Random() };
        w = DenseMatrix.CreateRandom(inputSize, outputSize, distribution);
        b = DenseMatrix.CreateRandom(inputSize, 1, distribution);
        output = DenseMatrix.Create(inputSize, 1, 1f);
    }

    public void Init(float w, float b)
    {
        this.w = DenseMatrix.Create(this.w.RowCount, this.w.ColumnCount, w); this.b = DenseMatrix.Create(this.w.RowCount, 1, b);
    }

    public DenseMatrix Forward(DenseMatrix input)
    {
        this.input = input;
        // w * input + b 即权重和，仅构成了该神经元的input，而output = activator(input)
        return output = activator.Forward(w * input + b);
    }

    public DenseMatrix Backward(DenseMatrix deltaFromNextLayer)
    {
        // 假设当前层为L层
        // L层的input就是L-1层的output
        // y = w · L-1层的output + b 对 w 求导，结果为L-1层的output
        // 这里的delta和wGradient是用向量表示，wGradient_i = delta_i * input，因此就是wGradient的每个分量乘上delta的每个分量
        // ********************注意*********************
        // 梯度由2部分组成：L-1层的output 和 L层的delta
        // wGradientL-1toL梯度（向量表示）为当前层的delta的各分量分别乘上inputL-1toL
        // 
        // deltaFromNextLayer是列矩阵（deltaFromNextLayer_i表示当前L层的第 i 个误差），
        // 右乘列矩阵input（input_j表示L-1层的第 j 个output）的转置即行矩阵
        // *****重点********：结果恰好为一个矩阵，其 i 行 j 列元素为L-1层的第 j 个output * 当前L层的第 i 个误差的结果
        wGradient = deltaFromNextLayer * input.Transpose() as DenseMatrix;
        //wGradient = DenseMatrix.op_DotMultiply(deltaFromNextLayer, input.Transpose()) as DenseMatrix;
        bGradient = deltaFromNextLayer;
        // delta是通用的，需要缓存起来，而这里通过反向迭代地更新delta，则不需要把delta作为全局变量来缓存
        // 使用L层的delta，乘上wx + b 对 x 的偏导 w 和output = 1 / (e^-x + 1)对 x 的偏导 output(1 - output)
        // 得到的结果为L-1层的delta（同理，L-1层的delta乘上L-2层的output即可得到L-1层的梯度，如此迭代）
        return DenseMatrix.op_DotMultiply(w.Transpose() * deltaFromNextLayer, activator.Backward(input)) as DenseMatrix;
    }

    public void UpdateWB(double rate)
    {
        //wGradient矩阵的 i 行 j 列元素为L-1层的第 j 个output * 当前L层的第 i 个误差的结果
        // w_ij 是表示L-1层的第 j 个w
        w += wGradient * rate;
        b += bGradient * rate;
    }
}
