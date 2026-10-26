// A stand-in for MATLAB's mex.h, with just the part of the MEX API that MRST's
// amgcl_matlab.cpp uses, so that its gateway compiles unchanged and is called
// from Python (amgcl_matlab_py.cpp). mxArray holds a dense double matrix, a
// sparse double matrix (CSC: Jc/Ir/Pr) or a 1x1 struct of such fields.
#pragma once
#include <cstddef>
#include <cstdarg>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

typedef std::size_t mwSize;
typedef std::size_t mwIndex;
enum mxComplexity { mxREAL = 0, mxCOMPLEX = 1 };

struct mxArray {
    std::size_t m = 0, n = 0;
    bool sparse = false;
    bool is_struct = false;
    std::vector<double> pr;
    std::vector<mwIndex> ir, jc;
    std::map<std::string, mxArray *> fields;
    ~mxArray() { for (auto &kv : fields) delete kv.second; }
};

inline std::size_t mxGetM(const mxArray *a) { return a->m; }
inline std::size_t mxGetN(const mxArray *a) { return a->n; }
inline bool mxIsDouble(const mxArray *a) { return !a->is_struct; }
inline bool mxIsComplex(const mxArray *) { return false; }
inline bool mxIsSparse(const mxArray *a) { return a->sparse; }
inline mxArray *mxCreateDoubleMatrix(std::size_t m, std::size_t n, mxComplexity) {
    mxArray *a = new mxArray();
    a->m = m; a->n = n; a->pr.assign(m*n, 0.0);
    return a;
}
inline double *mxGetPr(const mxArray *a) { return const_cast<double *>(a->pr.data()); }
inline mwIndex *mxGetIr(const mxArray *a) { return const_cast<mwIndex *>(a->ir.data()); }
inline mwIndex *mxGetJc(const mxArray *a) { return const_cast<mwIndex *>(a->jc.data()); }
inline std::size_t mxGetNzmax(const mxArray *a) { return a->pr.size(); }
inline double mxGetScalar(const mxArray *a) { return (a == nullptr || a->pr.empty()) ? 0.0 : a->pr[0]; }
inline mxArray *mxGetField(const mxArray *a, std::size_t, const char *name) {
    auto it = a->fields.find(name);
    return it == a->fields.end() ? nullptr : it->second;
}
inline int mexPrintf(const char *fmt, ...) {
    va_list args; va_start(args, fmt); int r = std::vprintf(fmt, args); va_end(args); return r;
}
[[noreturn]] inline void mexErrMsgTxt(const char *msg) { throw std::runtime_error(msg); }
[[noreturn]] inline void mexErrMsgIdAndTxt(const char *id, const char *fmt, ...) {
    char buf[1024];
    va_list args; va_start(args, fmt); std::vsnprintf(buf, sizeof(buf), fmt, args); va_end(args);
    throw std::runtime_error(std::string(id) + ": " + buf);
}
inline int mexAtExit(void (*)(void)) { return 0; }
