// #include <stdio.h>
// #include <math.h>
// #include <iostream>
// 
// int main()
// {
//     double x = 0;
//     clock_t start = clock(), diff;
//     for (double i = 0; i < 10000000; i++)
//     {
//         x = exp(i / 10000000);
//     }
//     diff = clock() - start;
//     int msec = diff * 1000 / CLOCKS_PER_SEC;
//     printf("Time taken %d seconds %d milliseconds", msec/1000, msec%1000);
//     std::cout << x << std::endl;
// }
// 
#include "catch_with_main.hpp"

unsigned int Factorial( unsigned int number ) {
    return number <= 1 ? number : Factorial(number-1)*number;
}

TEST_CASE( "Factorials are computed", "[factorial]" ) {
    REQUIRE( Factorial(1) == 1 );
    REQUIRE( Factorial(2) == 2 );
    REQUIRE( Factorial(3) == 6 );
    REQUIRE( Factorial(4) == 24 );
    REQUIRE( Factorial(10) == 3628800 );
}
