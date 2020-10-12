using MathNet.Numerics.LinearAlgebra.Double;

public class ReLUActivator : Activator
{
    public override DenseMatrix Forward(DenseMatrix input)
    {
        return (-input).PointwiseExp().Add(1).DivideByThis(1) as DenseMatrix;
    }

    public override DenseMatrix Backward(DenseMatrix output)
    {
        return DenseMatrix.op_DotMultiply(output, output.SubtractFrom(1)) as DenseMatrix;
    }
}
