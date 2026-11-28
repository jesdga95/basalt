// llama_project: run GGUF model(s) over token sequences, linked against an existing llama.cpp build.
//
// project: llama_project project model.gguf req.bin out.bin ngl [ubatch]
//   For every requested row (sequence, position, k ids) writes float lp[k] (full-vocab log-prob of
//   each id), float leftover (1 - sum p over the k ids), int32 top1 (argmax over the full vocab).
//
// check: llama_project check modelA.gguf modelB.gguf req.bin out.bin [ubatch]
//   Truncation-bias check inside one engine. A runs on GPU 0, B on GPU 1. For every row (only the
//   positions are used) computes KLD(A||B) over the full vocabulary and the same KLD with A's
//   truncated reference (smallest top set with cum prob >= 1-1e-6, cap 256, leftover as one bucket on
//   both sides). Writes per row: double kld_full, double kld_trunc[4] (caps 64,256,1024,4096), int32 same_top, int32 k[4].
#include "llama.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
static const int NC=4;
struct Row { int32_t pos; std::vector<int32_t> ids; };
struct Seq { std::vector<int32_t> toks; std::vector<Row> rows; };
static void rd(FILE*f, void*p, size_t n){ if(fread(p,1,n,f)!=n){fprintf(stderr,"short read\n");exit(1);} }

struct Ctx { llama_model*m; llama_context*c; llama_batch b; int nv; int ub; };
static Ctx open_model(const char*path,int ngl,int gpu,int maxlen,int ub){
  Ctx x; x.ub=ub;
  auto mp=llama_model_default_params(); mp.n_gpu_layers=ngl;
  if(gpu>=0){ mp.main_gpu=gpu; mp.split_mode=LLAMA_SPLIT_MODE_NONE; }
  x.m=llama_model_load_from_file(path,mp); if(!x.m){fprintf(stderr,"load fail %s\n",path);exit(1);}
  auto cp=llama_context_default_params(); cp.n_ctx=maxlen; cp.n_batch=ub; cp.n_ubatch=ub; cp.n_seq_max=1;
  x.c=llama_init_from_model(x.m,cp); if(!x.c){fprintf(stderr,"ctx fail\n");exit(1);}
  x.nv=llama_vocab_n_tokens(llama_model_get_vocab(x.m));
  x.b=llama_batch_init(ub,0,1);
  return x;
}
static void decode(Ctx&x,const std::vector<int32_t>&t,int st,int len){
  x.b.n_tokens=len;
  for(int j=0;j<len;++j){x.b.token[j]=t[st+j];x.b.pos[j]=st+j;x.b.n_seq_id[j]=1;x.b.seq_id[j][0]=0;x.b.logits[j]=1;}
  if(llama_decode(x.c,x.b)){fprintf(stderr,"decode fail\n");exit(1);}
}
static void logsm(const float*lg,int nv,std::vector<double>&lp,int&am){
  float mx=lg[0]; am=0; for(int i=1;i<nv;++i) if(lg[i]>mx){mx=lg[i];am=i;}
  double se=0; for(int i=0;i<nv;++i) se+=std::exp((double)lg[i]-mx);
  double lse=mx+std::log(se); lp.resize(nv); for(int i=0;i<nv;++i) lp[i]=(double)lg[i]-lse;
}

int main(int argc,char**argv){
  std::string mode=argc>1?argv[1]:"";
  bool chk=mode=="check";
  if((mode=="project"&&argc<7)||(chk&&argc<6)||(!chk&&mode!="project")){fprintf(stderr,"see header for usage\n");return 1;}
  const char*reqp=chk?argv[4]:argv[3]; const char*outp=chk?argv[5]:argv[4];
  int ub=chk?(argc>6?atoi(argv[6]):512):(argc>6?atoi(argv[6]):2048);
  FILE*f=fopen(reqp,"rb"); int32_t ns; rd(f,&ns,4);
  std::vector<Seq> seqs(ns); int maxlen=0;
  for(auto&s:seqs){int32_t n;rd(f,&n,4);s.toks.resize(n);rd(f,s.toks.data(),4*(size_t)n);maxlen=std::max(maxlen,(int)n);
    int32_t nr;rd(f,&nr,4);s.rows.resize(nr);
    for(auto&r:s.rows){int32_t k;rd(f,&r.pos,4);rd(f,&k,4);r.ids.resize(k);rd(f,r.ids.data(),4*(size_t)k);}}
  fclose(f);
  llama_backend_init();
  Ctx A=open_model(argv[2],99,chk?0:-1,maxlen,ub);
  Ctx B; if(chk) B=open_model(argv[3],99,1,maxlen,ub);
  FILE*o=fopen(outp,"wb");
  size_t done=0,total=0; for(auto&s:seqs) total+=s.rows.size();
  for(size_t si=0;si<seqs.size();++si){
    auto&s=seqs[si]; llama_memory_clear(llama_get_memory(A.c),true); if(chk) llama_memory_clear(llama_get_memory(B.c),true);
    int n=s.toks.size(); size_t ri=0;
    for(int st=0;st<n;st+=ub){
      int len=std::min(ub,n-st);
      decode(A,s.toks,st,len); if(chk) decode(B,s.toks,st,len);
      size_t r0=ri; while(ri<s.rows.size()&&s.rows[ri].pos<st+len) ++ri;
      size_t nr=ri-r0;
      if(!chk){
        std::vector<std::vector<float>> out(nr); std::vector<float> lo(nr); std::vector<int32_t> t1(nr);
        #pragma omp parallel for schedule(dynamic,1)
        for(long q=0;q<(long)nr;++q){
          auto&r=s.rows[r0+q]; const float*lg=llama_get_logits_ith(A.c,r.pos-st);
          std::vector<double> lp; int am; logsm(lg,A.nv,lp,am);
          out[q].resize(r.ids.size()); double sp=0;
          for(size_t k=0;k<r.ids.size();++k){out[q][k]=(float)lp[r.ids[k]]; sp+=std::exp(lp[r.ids[k]]);}
          lo[q]=(float)std::max(0.0,1.0-sp); t1[q]=am;
        }
        for(size_t q=0;q<nr;++q){fwrite(out[q].data(),4,out[q].size(),o);fwrite(&lo[q],4,1,o);fwrite(&t1[q],4,1,o);}
      } else {
        std::vector<double> kf(nr),kt(nr*NC); std::vector<int32_t> same(nr),kk(nr*NC);
        #pragma omp parallel for schedule(dynamic,1)
        for(long q=0;q<(long)nr;++q){
          auto&r=s.rows[r0+q];
          std::vector<double> la,lb; int aa,ab;
          logsm(llama_get_logits_ith(A.c,r.pos-st),A.nv,la,aa);
          logsm(llama_get_logits_ith(B.c,r.pos-st),B.nv,lb,ab);
          double full=0; for(int i=0;i<A.nv;++i){double p=std::exp(la[i]); full+=p*(la[i]-lb[i]);}
          std::vector<int> ord(A.nv); std::iota(ord.begin(),ord.end(),0);
          std::partial_sort(ord.begin(),ord.begin()+4097,ord.end(),[&](int x,int y){return la[x]>la[y];});
          const int caps[NC]={64,256,1024,4096};
          for(int ci=0;ci<NC;++ci){
            double sa=0,sb=0,tr=0; int k=0;
            for(;k<caps[ci];++k){ int i=ord[k]; double pa=std::exp(la[i]); tr+=pa*(la[i]-lb[i]); sa+=pa; sb+=std::exp(lb[i]); if(sa>=1-1e-6){++k;break;} }
            double LA=std::max(0.0,1.0-sa),LB=std::max(0.0,1.0-sb);
            if(LA>0&&LB>0) tr+=LA*std::log(LA/LB);
            kt[q*NC+ci]=tr; kk[q*NC+ci]=k;
          }
          kf[q]=full; same[q]=(aa==ab);
        }
        for(size_t q=0;q<nr;++q){fwrite(&kf[q],8,1,o);fwrite(&kt[q*NC],8,NC,o);fwrite(&same[q],4,1,o);fwrite(&kk[q*NC],4,NC,o);}
      }
      done+=nr;
    }
    fprintf(stderr,"seq %zu/%zu rows %zu/%zu\n",si+1,seqs.size(),done,total);
  }
  fclose(o); return 0;
}
