// --------------------------------------------------------------------------------------------------------------------
// <summary>
//   Defines the Program type.
// </summary>
// --------------------------------------------------------------------------------------------------------------------

namespace ProbabilisticProjections
{
    using System.Linq;

    using MicrosoftResearch.Infer.Maths;
    using MicrosoftResearch.Infer.Models;

    /// <summary>
    /// The program.
    /// </summary>
    public class Program
    {
        /// <summary>
        /// Defines the entry point of the application.
        /// </summary>
        public static void Main()
        {
            // Number of views.
            var P = Variable.New<int>().Named("P");
            Range p = new Range(P).Named("p");

            // Dimension of the shared latent vector y0
            var Q0 = Variable.New<int>().Named("q0");
            Range q0 = new Range(Q0).Named("q0");
            var zero = Variable.Array<double>(q0).Named("zero");
            zero[p] = 0.0;

            // Dimensions of each view specific latent vector yp
            var Qp = Variable.Array<int>(p).Named("Qp");
            Range qp = new Range(Qp[p]).Named("qp");
            var zeros = Variable.Array(Variable.Array<double>(qp), p).Named("zeros");
            zeros[p][qp] = 0.0;

            // Number of examples
            var N = Variable.New<int>().Named("N");
            Range n = new Range(N).Named("n");
            
            // Shared latent vector y0
            var y0priorPrecision = Variable.New<PositiveDefiniteMatrix>().Named("v0priorPrecision");
            var y0 = Variable.VectorGaussianFromMeanAndPrecision(Variable.Vector(zero).Named("y0PriorMean"), y0priorPrecision).Named("y0Prior");

            // View specific latent vectors yp
            var y0PriorMeans = Variable.Array<Vector>(p).Named("ypPriorMeans");
            y0PriorMeans[p] = Variable.Vector(zeros[p]).Named("zerosp");
            var ypPriorPrecision = Variable.Array<PositiveDefiniteMatrix>(p).Named("ypPriorPrecisions");
            var yp = Variable.Array<Vector>(p).Named("yp");
            yp[p] = Variable.VectorGaussianFromMeanAndPrecision(y0PriorMeans[p], ypPriorPrecision[p]);

            var V = Variable.Array<Matrix>(p).Named("V");
            var W = Variable.Array<Matrix>(p).Named("W");

            // todo: set the priors on V and W (zero mean Gaussian on each entry)

            var mu = Variable.Array<Vector>(p).Named("mu");
            // mu[p] = Variable.VectorGaussianFromMeanAndPrecision();

            var epsilonPrecision = Variable.Array<PositiveDefiniteMatrix>(p).Named("epsilonPrecision");
            var epsilon = Variable.Array<Vector>(p).Named("epsilon");
            epsilon[p] = Variable.VectorGaussianFromMeanAndPrecision(Variable.Vector(zeros[p]), epsilonPrecision[p]);

            var x = Variable.Array(Variable.Array<Vector>(p), n).Named("x");

            using (Variable.ForEach(n))
            {
                using (Variable.ForEach(p))
                {
                    var ap = Variable.MatrixTimesVector(V[p], yp[p]).Named("ap");
                    var a0 = Variable.MatrixTimesVector(W[p], y0).Named("a0");

                    x[n][p] = Variable.VectorGaussianFromMeanAndPrecision(ap + a0 + mu[p], epsilonPrecision[p]);
                }
            }

            // todo: 
            // how to set the prior precision matrices?
            // how to set tau_p, the parameter of the precision on epsilon?

            // Two views
            P.ObservedValue = 2;

            // 10 data points
            N.ObservedValue = 10;

            // Shared latent vector 5D
            Q0.ObservedValue = 5;

            // View specific latent vectors 5D
            Qp.ObservedValue = new[] { 5, 5 };

            y0priorPrecision.ObservedValue = PositiveDefiniteMatrix.Identity(Q0.ObservedValue);
            ypPriorPrecision.ObservedValue = Enumerable.Repeat(PositiveDefiniteMatrix.Identity(Q0.ObservedValue), 2).ToArray();

            
        }
    }
}
