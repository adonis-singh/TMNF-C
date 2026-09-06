/* Directed crossings through the public CPU/CUDA environments. */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "world.h"
#include "tmnf_cuda_env.h"
static void equal(const void *a,const void *b,size_t n,const char *what,unsigned gate,unsigned tick) {
 if(memcmp(a,b,n)) { fprintf(stderr,"%s mismatch gate %u tick %u\n",what,gate,tick);exit(1); }
}
int main(int argc,char **argv) {
 if(argc!=5)return 2;
 uint8_t hash[32];for(int i=0;i<32;i++){unsigned x;if(sscanf(argv[4]+2*i,"%2x",&x)!=1)return 2;hash[i]=x;}
 TmnfTrack *track=TmnfTrack_Load(argv[1],hash);TmnfRoute *route=TmnfRoute_Load(argv[3],hash);
 TmnfWorld *owner=World_Create(track,argv[2]);TmnfPhysicsWorld *world=World_GetPhysicsWorld(owner);uint32_t index=0;
 TmnfVecEnvConfig cfg=TmnfVecEnv_DefaultConfig();cfg.action_space=TMNF_ACTION_SPACE_ANALOG;cfg.autoreset_mode=TMNF_AUTORESET_NEXT_STEP;
 TmnfVecEnv cpu;TmnfVecEnv_Init(&cpu,&world,&index,1,route,&cfg);
 TmnfCudaVecEnvLimits limits=TmnfCudaVecEnv_DefaultLimits();TmnfCudaVecEnv *gpu=TmnfCudaVecEnv_Create(track,owner,route,1,&cfg,&limits);
 if(!gpu){fprintf(stderr,"%s\n",TmnfCudaVecEnv_LastError());return 1;}
 for(uint32_t gate=0;gate<route->metadata->finish_count;gate++) for(int ready=0;ready<2;ready++) {
  TmnfVecEnv_Reset(&cpu,NULL,NULL);TmnfEnvSnapshot snapshot;TmnfVecEnv_Capture(&cpu,&snapshot);
  const TmnfRouteTrigger *f=&route->finish[gate];int axis=0;
  const float *half=(const float*)&f->box.half_extent;
  if(half[1]<half[axis])axis=1;if(half[2]<half[axis])axis=2;
  GmVec3 position=f->box.center;((float*)&position)[axis]-=half[axis]+6;
  GmVec3_Mult_Iso4(&position,&f->transform);
  GmVec3 velocity={f->transform.m[axis]*40,f->transform.m[3+axis]*40,f->transform.m[6+axis]*40};
  CHmsStateDyna *states[]={&snapshot.live_state,&snapshot.committed_state,&snapshot.temp_state};
  for(auto state:states){state->pos=position;state->linVel=velocity;memcpy(state->tail,&velocity,sizeof(velocity));state->linVelAdded={};state->angVel={};state->force={};state->torque={};}
  snapshot.race_state.trigger_contacts=0;
  if(ready){snapshot.race_state.visited_checkpoints=(UINT64_C(1)<<route->metadata->checkpoint_count)-1;snapshot.race_state.visited_count=route->metadata->checkpoint_count;snapshot.race_state.passed_race_checkpoints=route->metadata->checkpoint_count;}
  TmnfVecEnv_Restore(&cpu,&snapshot);TmnfCudaVecEnv_Restore(gpu,&snapshot);
  TmnfAnalogAction action={};action.gas=1;bool finished=false;
  for(unsigned tick=0;tick<50;tick++) {
   TmnfStepResult a,b;TmnfVecEnv_StepAnalog(&cpu,&action,1,&a);TmnfCudaVecEnv_StepAnalog(gpu,&action,1,&b);
   equal(&a,&b,sizeof(a),"results",gate,tick);
   TmnfEnvSnapshot x,y;TmnfVecEnv_Capture(&cpu,&x);TmnfCudaVecEnv_Capture(gpu,&y);equal(&x,&y,sizeof(x),"snapshots",gate,tick);
   if(a.termination_reason==TMNF_TERMINATION_FINISH){finished=true;break;}
   if(a.terminated||a.truncated)break;
  }
  if(finished!=(ready||route->metadata->checkpoint_count==0)){fprintf(stderr,"gate %u ready %d finished %d\n",gate,ready,finished);return 1;}
  printf("gate %u block %u ready %d: exact CPU/CUDA crossing passed\n",gate,f->block_index,ready);
 }
 TmnfCudaVecEnv_Destroy(gpu);TmnfVecEnv_Destroy(&cpu);World_Destroy(owner);TmnfRoute_Unload(route);TmnfTrack_Unload(track);
}
