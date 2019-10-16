/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */


#include "cuda.h"
#include "../common/book.h"
#include "../common/cpu_anim.h"

#define DIM 1024
#define PI 3.1415926535897932f
#define MAX_TEMP 1.0f
#define MIN_TEMP 0.0001f
#define SPEED   0.25f

// these exist on the GPU side
texture<float,2>  texConstSrc;
texture<float,2>  texIn;
texture<float,2>  texOut;

__global__ void blend_kernel( float *dst,
                              bool dstOut ) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float   t, l, c, r, b;
    if (dstOut) {
        t = tex2D(texIn,x,y-1);
        l = tex2D(texIn,x-1,y);
        c = tex2D(texIn,x,y);
        r = tex2D(texIn,x+1,y);
        b = tex2D(texIn,x,y+1);
    } else {
        t = tex2D(texOut,x,y-1);
        l = tex2D(texOut,x-1,y);
        c = tex2D(texOut,x,y);
        r = tex2D(texOut,x+1,y);
        b = tex2D(texOut,x,y+1);
    }
    dst[offset] = c + SPEED * (t + b + r + l - 4 * c);
}

__global__ void copy_const_kernel( float *iptr ) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float c = tex2D(texConstSrc,x,y);
    if (c != 0)
        iptr[offset] = c;
}

// globals needed by the update routine
struct DataBlock {
    unsigned char   *output_bitmap;
    float           *dev_inSrc;
    float           *dev_outSrc;
    float           *dev_constSrc;
    CPUAnimBitmap  *bitmap;

    cudaEvent_t     start, stop;
    float           totalTime;
    float           frames;
};

/**
 * Draw Triangle function from Spektre's Example on stackoverflow
 * https://stackoverflow.com/a/39062479
 */
void _troj_line(int *pl,int *pr,int x0,int y0,int x1,int y1)
{
    int *pp;
    int x,y,kx,ky,dx,dy,k,m,p;
    // DDA variables (d)abs delta,(k)step direction
    kx=0; dx=x1-x0; if (dx>0) kx=+1;  if (dx<0) { kx=-1; dx=-dx; }
    ky=0; dy=y1-y0; if (dy>0) ky=+1;  if (dy<0) { ky=-1; dy=-dy; }
    // target buffer according to ky direction
    if (ky>0) pp=pl; else pp=pr;
    // integer DDA line start point
    x=x0; y=y0;
    // fix endpoints just to be sure (wrong division constants by +/-1 can cause that last point is missing)
    pp[y1]=x1; pp[y0]=x0;
    if (dx>=dy) { // x axis is major
        k=dy+dy;
        m=(dy-dx); m+=m;
        p=m;
        for (;;)
            {
            pp[y]=x;
            if (x==x1) break;
            x+=kx;
            if (p>0) { y+=ky; p+=m; } else p+=k;
            }
    }
    else {       // y axis is major
        k=dx+dx;
        m=(dx-dy); m+=m;
        p=m;
        for (;;) {
            pp[y]=x;
            if (y==y1) break;
            y+=ky;
            if (p>0) { x+=kx; p+=m; } else p+=k;
        }
    }
}

void troj(int x0, int y0, int x1, int y1, int x2, int y2, int temp, float* matrix) {
    int *pl,*pr;        // left/right buffers
    pl=new int[DIM];
    pr=new int[DIM];
    int x,y,yy0,yy1,xx0,xx1;
    // boundary line coordinates to buffers
    _troj_line(pl,pr,x0,y0,x1,y1);
    _troj_line(pl,pr,x1,y1,x2,y2);
    _troj_line(pl,pr,x2,y2,x0,y0);
    // y range
    yy0=y0; if (yy0>y1) yy0=y1; if (yy0>y2) yy0=y2;
    yy1=y0; if (yy1<y1) yy1=y1; if (yy1<y2) yy1=y2;
    // fill with horizontal lines
    for (y=yy0;y<=yy1;y++) {
        if (pl[y]<pr[y]) { xx0=pl[y]; xx1=pr[y]; }
        else             { xx1=pl[y]; xx0=pr[y]; }
        for (x=xx0;x<=xx1;x++)
            matrix[x+y*DIM]=temp;
    }
    delete[] pl;
    delete[] pr;
}

/**
 * End of Spektre's code example
 */

void triangleFan(int* xpoints, int* ypoints, int points, int temp, float* matrix) {
    for (int tri = 0; tri < points - 2; tri++) {
        troj(xpoints[0], ypoints[0],
             xpoints[tri + 1], ypoints[tri + 1],
             xpoints[tri + 2], ypoints[tri + 2],
             temp,
             matrix);
    }
}


void anim_gpu( DataBlock *d, int ticks ) {
    HANDLE_ERROR( cudaEventRecord( d->start, 0 ) );
    dim3    blocks(DIM/16,DIM/16);
    dim3    threads(16,16);
    CPUAnimBitmap  *bitmap = d->bitmap;

    // since tex is global and bound, we have to use a flag to
    // select which is in/out per iteration
    volatile bool dstOut = true;
    for (int i=0; i<90; i++) {
        float   *in, *out;
        if (dstOut) {
            in  = d->dev_inSrc;
            out = d->dev_outSrc;
        } else {
            out = d->dev_inSrc;
            in  = d->dev_outSrc;
        }
        copy_const_kernel<<<blocks,threads>>>( in );
        blend_kernel<<<blocks,threads>>>( out, dstOut );
        dstOut = !dstOut;
    }
    float_to_color<<<blocks,threads>>>( d->output_bitmap,
                                        d->dev_inSrc );

    HANDLE_ERROR( cudaMemcpy( bitmap->get_ptr(),
                              d->output_bitmap,
                              bitmap->image_size(),
                              cudaMemcpyDeviceToHost ) );

    HANDLE_ERROR( cudaEventRecord( d->stop, 0 ) );
    HANDLE_ERROR( cudaEventSynchronize( d->stop ) );
    float   elapsedTime;
    HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime,
                                        d->start, d->stop ) );
    d->totalTime += elapsedTime;
    ++d->frames;
    printf( "Average Time per frame:  %3.1f ms\n",
            d->totalTime/d->frames  );
}

// clean up memory allocated on the GPU
void anim_exit( DataBlock *d ) {
    cudaUnbindTexture( texIn );
    cudaUnbindTexture( texOut );
    cudaUnbindTexture( texConstSrc );
    HANDLE_ERROR( cudaFree( d->dev_inSrc ) );
    HANDLE_ERROR( cudaFree( d->dev_outSrc ) );
    HANDLE_ERROR( cudaFree( d->dev_constSrc ) );

    HANDLE_ERROR( cudaEventDestroy( d->start ) );
    HANDLE_ERROR( cudaEventDestroy( d->stop ) );
}


int main( void ) {
    DataBlock   data;
    CPUAnimBitmap bitmap( DIM, DIM, &data );
    data.bitmap = &bitmap;
    data.totalTime = 0;
    data.frames = 0;
    HANDLE_ERROR( cudaEventCreate( &data.start ) );
    HANDLE_ERROR( cudaEventCreate( &data.stop ) );

    int imageSize = bitmap.image_size();

    HANDLE_ERROR( cudaMalloc( (void**)&data.output_bitmap,
                               imageSize ) );

    // assume float == 4 chars in size (ie rgba)
    HANDLE_ERROR( cudaMalloc( (void**)&data.dev_inSrc,
                              imageSize ) );
    HANDLE_ERROR( cudaMalloc( (void**)&data.dev_outSrc,
                              imageSize ) );
    HANDLE_ERROR( cudaMalloc( (void**)&data.dev_constSrc,
                              imageSize ) );

    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    HANDLE_ERROR( cudaBindTexture2D( NULL, texConstSrc,
                                   data.dev_constSrc,
                                   desc, DIM, DIM,
                                   sizeof(float) * DIM ) );

    HANDLE_ERROR( cudaBindTexture2D( NULL, texIn,
                                   data.dev_inSrc,
                                   desc, DIM, DIM,
                                   sizeof(float) * DIM ) );

    HANDLE_ERROR( cudaBindTexture2D( NULL, texOut,
                                   data.dev_outSrc,
                                   desc, DIM, DIM,
                                   sizeof(float) * DIM ) );

    float *temp = (float*)malloc( imageSize );

    int points = 4;
    int xpoints[points] = {96, 48, 208, 204};
    int ypoints[points] = {740, 244, 204, 560};
    triangleFan(xpoints, ypoints, points, MAX_TEMP, temp);

    int points2 = 4;
    int xpoints2[points] = {96, 252, 548, 408};
    int ypoints2[points] = {740, 768, 208, 164};
    triangleFan(xpoints2, ypoints2, points2, MAX_TEMP, temp);

    int points3 = 4;
    int xpoints3[points] = {548, 624, 484, 452};
    int ypoints3[points] = {208, 696, 716, 388};
    triangleFan(xpoints3, ypoints3, points3, MAX_TEMP, temp);

    HANDLE_ERROR( cudaMemcpy( data.dev_constSrc, temp,
                              imageSize,
                              cudaMemcpyHostToDevice ) );    

    HANDLE_ERROR( cudaMemcpy( data.dev_inSrc, temp,
                              imageSize,
                              cudaMemcpyHostToDevice ) );
    free( temp );

    bitmap.anim_and_exit( (void (*)(void*,int))anim_gpu,
                           (void (*)(void*))anim_exit );
}

