// This is the convolutiona function in C
#include<stdio.h>

void conv (float* op, float* x, float* k, int x_row, int x_col, int k_row, int k_col);

void conv (float* op, float* x, float* k, int x_row, int x_col, int k_row, int k_col) {
        
      int conv_row = x_row-k_row+1;
      int conv_col = x_col-k_col+1;
      
      //for addressing the output image
      int row, col;
      
      //for addressing the kernel
      int i, j;
      
      //for addressing the input image
      int row_temp, col_temp;
      
      for (row = 0; row <= (conv_row-1); row++) {
         for (col = 0; col <= (conv_col-1); col++) { 
            *(op + (row * conv_col) + col) = 0;
            for (i = k_row-1; i >= 0; i--) {
               for (j = k_col-1; j >= 0; j--) {
        	       row_temp = row + k_row - (i+1);
        	       col_temp = col + k_col - (j+1);
        	       *(op + (row * conv_col + col)) += (*(x + (row_temp*x_col + col_temp))) * (*(k + (i*k_col + j)));
               }
            }
         }
      }

      return;
}



