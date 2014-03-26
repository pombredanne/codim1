delta_chi = 1./( PI4*r ); 
fac[0] = (1+nu)/(8*PI*E*(1-nu));
fac[2] = -1/(8*PI*(1-nu));
const double mue      = - fac[2]/(2.*fac[0]); 
const double mue2t4   = 4.*mue*mue;
for(unsigned int i=0; i<3; ++i)
    for(unsigned int j=i; j<3; ++j)
        fund_sol_U(i,j) = 2.*mue*delta_chi*KD(i,j) - mue2t4*fund_sol_U(i,j); 

fund_sol_U(1,0) = fund_sol_U(0,1);                    
fund_sol_U(2,0) = fund_sol_U(0,2);                    
fund_sol_U(2,1) = fund_sol_U(1,2);

// combinations of curl_x[ i ] and curl_y[ j ] (outer product)
const double SXY00 = curl_x[cx_i][0]*curl_y[cy_j][0];
const double SXY01 = curl_x[cx_i][0]*curl_y[cy_j][1];
const double SXY10 = curl_x[cx_i][1]*curl_y[cy_j][0];
const double SXY11 = curl_x[cx_i][1]*curl_y[cy_j][1];

// initialize result array
for(unsigned int i=0; i < 2; ++i)
        for(unsigned int j=0; j<2; ++j)
                result_curl(cx_i,cy_j)(i,j) = 0.;


// evaluate part1
const double factor = 2.*delta_chi*mue*(SXY22);
for(unsigned int i=0; i<3; ++i)
        result_curl(cx_i,cy_j)(i,i) = factor;


// evaluate part 2
result_curl(cx_i,cy_j)(0,0) -= delta_chi*mue*SXY00;
result_curl(cx_i,cy_j)(0,1) -= delta_chi*mue*SXY01;
result_curl(cx_i,cy_j)(1,0) -= delta_chi*mue*SXY10;
result_curl(cx_i,cy_j)(1,1) -= delta_chi*mue*SXY11;

const double factor = 2.*delta_chi*mue*(SXY00 + SXY11);
result_curl(cx_i,cy_j)(2,2) += 
        SXY11 * fund_sol_U(0,0) - SXY10 * fund_sol_U(0,1) - 
        SXY01 * fund_sol_U(1,0) + SXY00 * fund_sol_U(1,1);
