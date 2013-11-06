#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;

// function declarations
vec forward_euler(vec v, double alpha);
vec backward_euler(vec v, double alpha);
vec crank_nicolson(vec v, double alhpa);
vec tridiag(double a, double b, double c, vec vnew, vec v, int n);


int main(int argc, char* argv[])
{
    // initialization
    int n = 99;                  // number of spatial steps (n+1 = 10 mesh points in x)
    double dx = 1.0/(n+1);      // spatial step (dx = 0.01)
    cout << dx << endl;
    double dt = 0.5*dx*dx;      // time step (dt = 0.00005)
    double alpha = dt/(dx*dx);
    double t_fin = 1.0;         // final time
    int tsteps = t_fin/dt;      // number of time steps (tsteps = 200)

    // decide method
    int method = atoi(argv[1]);

    // make system state vector v. vj contains all x-values at time tj
    vec v = linspace<vec>(0, 1, n+1);
    vec vnew = zeros<vec>(n+1);

    // initial condition
    v = v - 1.0;

    // boundary conditions
    v(0) = v(n)= 0.0;

    // write out initial v
    fstream outFile;
    if (method == 1) outFile.open("data1.dat", ios::out);
    if (method == 2) outFile.open("data2.dat", ios::out);
    if (method == 3) outFile.open("data3.dat", ios::out);

    for (int i=0; i<=n; i++) {
        outFile << v(i) << " ";
    }
    outFile << endl;

    // time integration
    for (int t=1; t <= tsteps; t++) {
        if (method == 1) vnew = forward_euler(v, alpha);
        if (method == 2) vnew = backward_euler(v, alpha);
        if (method == 3) vnew = crank_nicolson(v, alpha);

        // write out system state vector
        for (int i=0; i<=n; i++) {
            outFile << vnew(i) << " ";
        }
        outFile << endl;

        // make ready for next iteration
        v = vnew;

    }
    outFile.close();

    return 0;
}


// computes vnew using the forward Euler scheme, equation: vnew = Av
vec forward_euler(vec v, double alpha)
{
    int n = v.n_elem - 1;
    vec vnew = zeros<vec>(n+1);

    // boundary conditions
    vnew(0) = vnew(n) = 0.0;

    // vector-matrix multiplication, boundaries doesn't change
    for (int i=1; i < n; i++) {
        vnew(i) = alpha*v(i-1) + (1 - 2*alpha)*v(i) + alpha*v(i+1);
    }

    return vnew;
}


// computes vnew using the backward Euler shceme, equation: Avnew = v
vec backward_euler(vec v, double alpha)
{
    int n = v.n_elem - 1;
    vec vnew = zeros<vec>(n+1);
    double a, b, c;

    // boundary conditions
    vnew(0) = vnew(n) = 0.0;

    // set diagonal and sub- and superdiagonal of tridiagonal matrix A
    a = c = - alpha;
    b = 1 + 2*alpha;

    // row-reduce to solve equation
    vnew = tridiag(a, b, c, vnew, v, n);

    return vnew;
}


// computes vnew using the Crank-Nicolson scheme, equation: Avnew = Bv
// this equation can be seperated into two steps: 1. r = Bv, 2. Avnew = r
vec crank_nicolson(vec v, double alpha)
{
    int n = v.n_elem - 1;
    vec vnew = zeros<vec>(n+1);
    vec r = zeros<vec>(n+1);
    double a, b, c;

    // boundary contions
    vnew(0) = r(0) = vnew(n) = r(n) = 0.0;

    // step 1: vector-matrix multiplication, boundaries doesn't change
    for (int i=1; i<n; i++) {
        r(i) = alpha*v(i-1) + (2 - 2*alpha)*v(i) + alpha*v(i+1);
    }

    // set diagonal and sub- and superdiagonal of tridiagonal matrix A
    a = c = - alpha;
    b = 2 + 2*alpha;


    // step 2: row-reduce to solve equation
    vnew = tridiag(a, b, c, vnew, r, n);
    //vnew.print();
    return vnew;
}


// function to solve equation Ax = y by row-reducing when A is a tridiagonal matrix
vec tridiag(double da, double db, double dc, vec vnew, vec v, int n)
{

    // initialize a, b and c as vectors
    vec a = zeros<vec>(n+1); vec b = zeros<vec>(n+1); vec c = zeros<vec>(n+1);
    a.fill(da); c.fill(dc); b.fill(db);

    // forward substitution
    for (int i=0; i<n; i++) {

        b(i+1) = b(i+1) - (a(i)*c(i))/b(i);
        v(i+1) = v(i+1) - (a(i)*v(i))/b(i);
    }

    // first step on backward substituion
    vnew(n-1) = v(n-1)/b(n-1);

    // backward substitution
    for (int i=n-2; i>=1; i--) {
        vnew(i) = (v(i) - c(i)*vnew(i+1))/b(i);
    }
    //vnew.print();
    return vnew;
}
