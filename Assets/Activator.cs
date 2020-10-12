using MathNet.Numerics.LinearAlgebra.Double;

public abstract class Activator
{
    public abstract DenseMatrix Forward(DenseMatrix input);

    public abstract DenseMatrix Backward(DenseMatrix output);
}
