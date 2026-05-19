///////////////////////////////////////////////////////////////////////////
// Example code from the Udacity tutorial on CUDA                        //
// Link to the video here: https://www.youtube.com/watch?v=GiGE3QjwknQ   //
// TO COMPILE: $ nvcc -o square square.cu                                //
///////////////////////////////////////////////////////////////////////////

//
// NOTES:
//  * Device: is the term of the GPU
//  * Host: is the term of the CPU
//  * kernels are the only things run in parallel on the GPU
//  * everything in the main is run on CPU
//  * memory transfers between Host (CPU) and Device (GPU) should be minimal
//  * kernels all run at the same time
//  * threads can know their Id's with threadIdx.x, blocks are similar
//

#include <stdio.h>

// =========================================================================
// 1. GPU 커널 정의
// =========================================================================
// __global__: "이 함수는 CPU가 호출하지만, 실제 실행은 GPU가 한다"는 뜻
// *d_in, *d_out: GPU 메모리에 올라간 Tensor들의 주소값
__global__ void square(float *d_out, float *d_in){
    // threadIdx.x: 1024개의 스레드들이 각자 부여받은 0~1023번의 고유 ID
    int idx = threadIdx.x; 
    // [메모리 읽기] 각 스레드가 GPU 메모리(d_in)에서 '자기 스레드 번호(idx)'에 해당하는 칸의 숫자 가져오기
    float f = d_in[idx];
    // [연산 및 쓰기] 꺼내온 숫자를 제곱(f*f)한 뒤, 결과 배열(d_out)의 '자기 칸(idx)'에 저장하기
    d_out[idx] = f*f; // 각자 자기 칸만 제곱하므로 스레드 간 충돌이 없음 (독립 구조)
}

// =========================================================================
// 2. 메인 함수 (여기서부터는 무조건 CPU가 순차적으로 실행)
// =========================================================================
int main(){
  const int ARRAY_SIZE = 1024; // 배열의 크기 (한 블록 당 최대 1024개의 스레드 실행 가능)
  const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float); // 메모리 할당을 위한 바이트 크기 계산

  // CPU 메모리에 배열 할당 및 초기화 (0.0, 1.0, 2.0, ..., 1023.0)
  float h_in[ARRAY_SIZE];
  for (int i = 0; i < ARRAY_SIZE; i++){
    h_in[i] = float(i);
  }
  // CPU 메모리에 출력 배열 생성
  float h_out[ARRAY_SIZE];
  
  // GPU 메모리에 할당될 배열을 가리키는 포인터 선언
  float *d_in;
  float *d_out;

  // GPU 메모리에 float 배열 공간을 생성하고, 해당 메모리 주소를 d_in, d_out 포인터에 저장
  cudaMalloc((void **) &d_in, ARRAY_BYTES);
  cudaMalloc((void **) &d_out, ARRAY_BYTES);

  // CPU 메모리(h_in)에 있는 데이터를 GPU 메모리(d_in)로 전달
  cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

  // GPU에서 square 커널 실행: 1개의 블록, 1024개의 스레드로 구성된 그리드에서 실행
  square<<<1,ARRAY_SIZE>>>(d_out,d_in);

  // GPU 메모리(d_out)에 있는 결과 데이터를 CPU 메모리(h_out)로 전달
  cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

  // GPU 계산 결과를 CPU에서 출력
  for (int i = 0; i < ARRAY_SIZE; i++){
    printf("%f", h_out[i]);
    printf(((i % 4) != 3) ? "\t" : "\n");
  }

  // GPU 메모리 해제
  cudaFree(d_in);
  cudaFree(d_out);

  return 0;
}