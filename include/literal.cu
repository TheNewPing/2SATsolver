#ifndef __LITERAL_CU__
#define __LITERAL_CU__

// literal.cu

#include <string>

struct Literal {
    unsigned int value;
    bool isPositive;

    // Constructor
    __host__ __device__ Literal(unsigned int v, bool isPos) : value(v), isPositive(isPos) {}

    // Constructor from string
    Literal(const std::string& str) {
        if (str[0] == '_') {
            isPositive = false;
            value = static_cast<unsigned int>(std::stoi(str.substr(1)));
        } else {
            isPositive = true;
            value = static_cast<unsigned int>(std::stoi(str));
        }
    }

    // to_string method
    __host__ std::string to_string() const {
        return isPositive ? std::to_string(value) : "_" + std::to_string(value);
    }

    // LessThanComparableConcept
    __host__ __device__ bool operator<(const Literal& other) const {
        return value < other.value;
    }
};

#endif // __LITERAL_CU__
