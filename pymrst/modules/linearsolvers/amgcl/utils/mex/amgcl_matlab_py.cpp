// Python gateway to MRST's amgcl_matlab MEX (amgcl_matlab.cpp, unchanged):
// builds the mxArrays MATLAB would pass, calls mexFunction, and returns its
// three outputs.
#include "mex.h"
#include "amgcl_matlab.cpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

static mxArray *scalar(double v) {
    mxArray *a = mxCreateDoubleMatrix(1, 1, mxREAL);
    a->pr[0] = v;
    return a;
}

static mxArray *column(py::array_t<double, py::array::c_style | py::array::forcecast> v) {
    mxArray *a = mxCreateDoubleMatrix(v.size(), 1, mxREAL);
    std::copy(v.data(), v.data() + v.size(), a->pr.begin());
    return a;
}

static mxArray *options_struct(const py::dict &opts) {
    mxArray *a = new mxArray();
    a->is_struct = true; a->m = 1; a->n = 1;
    for (auto item : opts) {
        std::string name = py::str(item.first);
        py::handle value = item.second;
        if (py::isinstance<py::bool_>(value) || py::isinstance<py::int_>(value) ||
            py::isinstance<py::float_>(value)) {
            a->fields[name] = scalar(value.cast<double>());
        } else {
            a->fields[name] = column(py::array_t<double>::ensure(value));
        }
    }
    return a;
}

static py::tuple amgcl_matlab(std::size_t n,
                              py::array_t<long long, py::array::c_style | py::array::forcecast> jc,
                              py::array_t<long long, py::array::c_style | py::array::forcecast> ir,
                              py::array_t<double, py::array::c_style | py::array::forcecast> pr,
                              py::array_t<double, py::array::c_style | py::array::forcecast> b,
                              const py::dict &opts, double tol, double maxit, double solver_id,
                              py::object reuse_id, py::object x0) {
    mxArray *A = new mxArray();
    A->m = n; A->n = n; A->sparse = true;
    A->jc.assign(jc.data(), jc.data() + jc.size());
    A->ir.assign(ir.data(), ir.data() + ir.size());
    A->pr.assign(pr.data(), pr.data() + pr.size());
    std::vector<const mxArray *> prhs = {A, column(b), options_struct(opts), scalar(tol),
                                         scalar(maxit), scalar(solver_id)};
    if (!reuse_id.is_none()) {
        prhs.push_back(scalar(reuse_id.cast<double>()));
        if (!x0.is_none()) prhs.push_back(column(py::array_t<double>::ensure(x0)));
    }
    mxArray *plhs[3] = {nullptr, nullptr, nullptr};
    try {
        mexFunction(3, plhs, (int)prhs.size(), prhs.data());
    } catch (...) {
        for (auto p : prhs) delete p;
        for (auto p : plhs) delete p;
        throw;
    }
    py::array_t<double> x(plhs[0]->m);
    std::copy(plhs[0]->pr.begin(), plhs[0]->pr.end(), x.mutable_data());
    py::tuple out = py::make_tuple(x, plhs[1]->pr[0], plhs[2]->pr[0]);
    for (auto p : prhs) delete p;
    for (auto p : plhs) delete p;
    return out;
}

PYBIND11_MODULE(_amgcl_matlab, m) {
    m.doc() = "MRST's amgcl_matlab MEX gateway, compiled for Python.";
    m.def("amgcl_matlab", &amgcl_matlab, py::arg("n"), py::arg("jc"), py::arg("ir"),
          py::arg("pr"), py::arg("b"), py::arg("opts"), py::arg("tol"), py::arg("maxit"),
          py::arg("solver_id"), py::arg("reuse_id") = py::none(), py::arg("x0") = py::none());
}
