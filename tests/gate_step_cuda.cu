#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "world.h"
#include "tmnf_cuda_env.h"
static void check(cudaError_t e) { if(e != cudaSuccess) { fprintf(stderr,"%s\n",cudaGetErrorString(e)); exit(1); } }
static void equal(const void *a,const void *b,size_t n,const char *what,int step) {
 if(memcmp(a,b,n)) { fprintf(stderr,"%s mismatch step %d\n",what,step); exit(1); }
}
int main(int argc,char **argv) {
 if(argc!=5) return 2;
 uint8_t hash[32]; for(int i=0;i<32;i++) { unsigned x; if(sscanf(argv[4]+2*i,"%2x",&x)!=1) return 2; hash[i]=x; }
 TmnfTrack *track=TmnfTrack_Load(argv[1],hash);
 TmnfRoute *route=TmnfRoute_Load(argv[3],hash);
 const int n=33;
 for(int analog=0;analog<2;analog++) for(int next=0;next<2;next++) {
  TmnfWorld *owners[n]; TmnfPhysicsWorld *worlds[n]; uint32_t indices[n]={0};
  for(int i=0;i<n;i++) { owners[i]=World_Create(track,argv[2]); worlds[i]=World_GetPhysicsWorld(owners[i]); }
  TmnfVecEnvConfig cfg=TmnfVecEnv_DefaultConfig(); cfg.thread_count=3; cfg.horizon_ticks=800;
  cfg.action_space=analog?TMNF_ACTION_SPACE_ANALOG:TMNF_ACTION_SPACE_DISCRETE;
  cfg.autoreset_mode=next?TMNF_AUTORESET_NEXT_STEP:TMNF_AUTORESET_SAME_STEP;
  TmnfVecEnv cpu; TmnfVecEnv_Init(&cpu,worlds,indices,n,route,&cfg);
  TmnfCudaVecEnvLimits limits=TmnfCudaVecEnv_DefaultLimits();
  TmnfCudaVecEnv *gpu=TmnfCudaVecEnv_Create(track,owners[0],route,n,&cfg,&limits);
  if(!gpu) { fprintf(stderr,"%s\n",TmnfCudaVecEnv_LastError());return 1; }
  TmnfGateObservations cg[n],cf[n],*gg,*gf; TmnfStepResult cr[n],gr[n];
  uint8_t *da; TmnfAnalogAction *aa;
  check(cudaMallocManaged(&gg,sizeof(cg))); check(cudaMallocManaged(&gf,sizeof(cf)));
  check(cudaMallocManaged(&da,n)); check(cudaMallocManaged(&aa,n*sizeof(*aa)));
  TmnfVecEnv_Reset(&cpu,NULL,NULL); TmnfCudaVecEnv_Reset(gpu,NULL,NULL);
  for(int step=0;step<200;step++) {
   for(int i=0;i<n;i++) { da[i]=(step+i)%12; aa[i]={}; aa[i].steer=((step+i)%3-1)*.5f; aa[i].gas=1; }
   if(analog) {
    TmnfVecEnv_StepAnalogWithGates(&cpu,aa,5,cr,cg,cf);
    TmnfCudaVecEnv_StepAnalogDeviceWithGates(gpu,aa,5,gg,gf);
   } else {
    TmnfVecEnv_StepDiscreteWithGates(&cpu,da,5,cr,cg,cf);
    TmnfCudaVecEnv_StepDiscreteDeviceWithGates(gpu,da,5,gg,gf);
   }
   check(cudaDeviceSynchronize());
   check(cudaMemcpy(gr,TmnfCudaVecEnv_DeviceResults(gpu),sizeof(gr),cudaMemcpyDeviceToHost));
   equal(cr,gr,sizeof(cr),"results",step); equal(cg,gg,sizeof(cg),"current gates",step); equal(cf,gf,sizeof(cf),"terminal gates",step);
   if(step%19==0) {
    TmnfEnvSnapshot snapshots[n],rotated[n]; TmnfVecEnv_Capture(&cpu,snapshots);
    for(int i=0;i<n;i++) rotated[i]=snapshots[(i+1)%n];
    TmnfVecEnv_Restore(&cpu,rotated); TmnfCudaVecEnv_Restore(gpu,rotated);
    TmnfVecEnv_ObserveGates(&cpu,cg); TmnfCudaVecEnv_ObserveGatesDevice(gpu,gg);
    check(cudaDeviceSynchronize()); equal(cg,gg,sizeof(cg),"cross-slot restored gates",step);
   }
  }
  check(cudaFree(aa));check(cudaFree(da));check(cudaFree(gf));check(cudaFree(gg));
  TmnfCudaVecEnv_Destroy(gpu);TmnfVecEnv_Destroy(&cpu);
  for(int i=0;i<n;i++) World_Destroy(owners[i]);
 }
 TmnfRoute_Unload(route);TmnfTrack_Unload(track);
 puts("gate step parity: 33 slots, both actions/autoresets, terminal and cross-slot restore passed");
}
