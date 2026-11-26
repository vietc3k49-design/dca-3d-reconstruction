#define EIGEN_DONT_PARALLELIZE
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <Eigen/SVD>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using V2=Eigen::Vector2d; using V3=Eigen::Vector3d;
using V6=Eigen::Matrix<double,6,1>;
using M3=Eigen::Matrix3d; using M6=Eigen::Matrix<double,6,6>;
using M23=Eigen::Matrix<double,2,3>; using M26=Eigen::Matrix<double,2,6>;
using M63=Eigen::Matrix<double,6,3>;
using Clock=std::chrono::steady_clock;
static double seconds(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
template<class T> static void readv(std::ifstream& f,T* p,size_t n){
    if(!f.read(reinterpret_cast<char*>(p),std::streamsize(n*sizeof(T)))) throw std::runtime_error("Truncated input");
}
template<class T> static void writev(std::ofstream& f,const T* p,size_t n){f.write(reinterpret_cast<const char*>(p),std::streamsize(n*sizeof(T)));}
struct Obs {int cam,point; V2 uv;};
struct State {std::vector<M3> R; std::vector<V3> C,X,O;};
struct Problem {
    int nc=0,np=0; double unit=.1;
    State initial; std::vector<std::array<double,4>> K;
    std::vector<M3> Rgt; std::vector<V3> Cgt;
    std::vector<Obs> obs,test; std::vector<int> ptr,forced;
    std::vector<V3> anchor_xyz;
};
static std::vector<M3> read_m3(std::ifstream& f,int n){
    std::vector<M3> out(n); for(auto& a:out)for(int r=0;r<3;r++)for(int c=0;c<3;c++)readv(f,&a(r,c),1); return out;
}
static std::vector<V3> read_v3(std::ifstream& f,int n){std::vector<V3> out(n); for(auto& a:out)readv(f,a.data(),3);return out;}
static std::vector<Obs> read_obs(std::ifstream& f,int n){
    std::vector<uint32_t> ci(n),pi(n); readv(f,ci.data(),n);readv(f,pi.data(),n);
    std::vector<Obs> out(n);for(int i=0;i<n;i++){out[i].cam=int(ci[i]);out[i].point=int(pi[i]);readv(f,out[i].uv.data(),2);}return out;
}
static Problem load(const std::string& path){
    std::ifstream f(path,std::ios::binary);if(!f)throw std::runtime_error("Cannot open input");
    char magic[8];readv(f,magic,8);if(std::string(magic,8)!="SDCABIN1")throw std::runtime_error("Wrong input format");
    uint32_t h[5];readv(f,h,5);Problem p;p.nc=int(h[0]);p.np=int(h[1]);
    if(p.nc<2||p.np<4||p.nc>10000||h[2]>100000000)throw std::runtime_error("Invalid input sizes");
    p.initial.R=read_m3(f,p.nc);p.initial.C=read_v3(f,p.nc);p.initial.X=read_v3(f,p.np);
    p.initial.O.assign(p.nc,V3::Zero());p.K.resize(p.nc);for(auto& k:p.K)readv(f,k.data(),4);
    p.obs=read_obs(f,int(h[2]));p.Rgt=read_m3(f,p.nc);p.Cgt=read_v3(f,p.nc);p.test=read_obs(f,int(h[3]));
    std::vector<uint32_t> ids(h[4]);readv(f,ids.data(),ids.size());for(auto j:ids)p.forced.push_back(int(j));
    p.anchor_xyz=read_v3(f,int(h[4]));readv(f,&p.unit,1);
    std::stable_sort(p.obs.begin(),p.obs.end(),[](const Obs&a,const Obs&b){return a.point<b.point||(a.point==b.point&&a.cam<b.cam);});
    p.ptr.assign(p.np+1,0);for(const auto&o:p.obs){if(o.point<0||o.point>=p.np||o.cam<0||o.cam>=p.nc)throw std::runtime_error("Invalid track");p.ptr[o.point+1]++;}
    for(int j=0;j<p.np;j++)p.ptr[j+1]+=p.ptr[j];return p;
}
static M3 skew(const V3&w){M3 S;S<<0.,-w[2],w[1],w[2],0.,-w[0],-w[1],w[0],0.;return S;}
static M3 exp3(const V3&w){
    double q=w.squaredNorm(),a=std::sqrt(q);
    double A=a<1e-4?1-q/6+q*q/120:std::sin(a)/a;
    double B=a<1e-4?.5-q/24+q*q/720:(1-std::cos(a))/q;M3 S=skew(w);return M3::Identity()+A*S+B*(S*S);
}
static M3 left_jac(const V3&w){
    double q=w.squaredNorm(),a=std::sqrt(q);
    double A=a<1e-4?.5-q/24+q*q/720:(1-std::cos(a))/q;
    double B=a<1e-4?1./6-q/120+q*q/5040:(a-std::sin(a))/(q*a);M3 S=skew(w);return M3::Identity()+A*S+B*(S*S);
}
struct Mask {std::vector<std::array<uint8_t,6>> cam;std::vector<uint8_t> point;};
static Mask mask_for(const Problem&p,const State&s,bool anchors){
    Mask m;m.cam.resize(p.nc);for(auto&a:m.cam)a.fill(1);m.point.assign(p.np,1);
    if(anchors){for(int j:p.forced)m.point[j]=0;}
    else {m.cam[0].fill(0);Eigen::Index dim;(s.C[1]-s.C[0]).cwiseAbs().maxCoeff(&dim);m.cam[1][3+dim]=0;}return m;
}
struct Plan {std::vector<uint8_t> core;std::vector<int> counts,perm,sizes;};
static void add_core(Plan& a,const Problem&p,int j){if(a.core[j])return;a.core[j]=1;for(int o=p.ptr[j];o<p.ptr[j+1];o++)a.counts[p.obs[o].cam]++;}
static uint32_t numpy_interval(uint32_t max,std::mt19937&gen){
    uint32_t mask=max;mask|=mask>>1;mask|=mask>>2;mask|=mask>>4;mask|=mask>>8;mask|=mask>>16;
    uint32_t value;do{value=gen()&mask;}while(value>max);return value;
}
static Plan make_plan(const Problem&p,uint32_t seed,bool full_only,int min_per_camera){
    Plan a;a.core.assign(p.np,0);a.counts.assign(p.nc,0);for(int j:p.forced)add_core(a,p,j);
    while(*std::min_element(a.counts.begin(),a.counts.end())<min_per_camera){
        int best=-1,score=-1;for(int j=0;j<p.np;j++){if(a.core[j])continue;int val=0;
            for(int o=p.ptr[j];o<p.ptr[j+1];o++)val+=std::max(0,min_per_camera-a.counts[p.obs[o].cam]);
            if(val>score){best=j;score=val;}}
        if(best<0||score<=0)throw std::runtime_error("Cannot cover every camera");add_core(a,p,best);
    }
    while(true){
        std::vector<int> parent(p.nc);std::iota(parent.begin(),parent.end(),0);
        auto root=[&](int k){while(parent[k]!=k)k=parent[k];return k;};
        for(int j=0;j<p.np;j++)if(a.core[j]&&p.ptr[j]<p.ptr[j+1]){
            int first=p.obs[p.ptr[j]].cam;for(int o=p.ptr[j];o<p.ptr[j+1];o++){int x=root(first),y=root(p.obs[o].cam);parent[y]=x;}}
        for(int k=0;k<p.nc;k++)parent[k]=root(k);
        if(std::all_of(parent.begin(),parent.end(),[&](int v){return v==parent[0];}))break;
        int best=-1,bc=1;for(int j=0;j<p.np;j++){if(a.core[j])continue;std::vector<uint8_t> seen(p.nc,0);int val=0;
            for(int o=p.ptr[j];o<p.ptr[j+1];o++){int r=parent[p.obs[o].cam];if(!seen[r]){seen[r]=1;val++;}}
            if(val>bc){best=j;bc=val;}}
        if(best<0)throw std::runtime_error("Disconnected camera graph");add_core(a,p,best);
    }
    for(int j=0;j<p.np;j++)if(!a.core[j])a.perm.push_back(j);
    std::mt19937 gen(seed);for(int i=int(a.perm.size())-1;i>0;i--){int j=int(numpy_interval(uint32_t(i),gen));std::swap(a.perm[i],a.perm[j]);}
    int core_count=int(std::count(a.core.begin(),a.core.end(),uint8_t(1)));
    if(full_only)a.sizes.push_back(p.np);else{
        std::array<int,6> proposed={std::max(core_count+8,32),96,256,768,2048,p.np};
        for(int size:proposed){size=std::min(p.np,std::max(core_count+1,size));if(a.sizes.empty()||size>a.sizes.back())a.sizes.push_back(size);}}
    return a;
}
struct Sample {std::vector<Obs> obs;std::vector<int> ptr;std::vector<uint8_t> selected;std::vector<double> weights;};
static Sample make_sample(const Problem&p,const Plan&plan,int size){
    Sample s;s.selected=plan.core;int core_count=int(std::count(plan.core.begin(),plan.core.end(),uint8_t(1)));
    int nr=size-core_count;for(int k=0;k<nr;k++)s.selected[plan.perm[k]]=1;
    s.weights.assign(p.np,0.);for(int j=0;j<p.np;j++)if(plan.core[j])s.weights[j]=1.;else if(s.selected[j])s.weights[j]=double(p.np-core_count)/nr;
    s.ptr.assign(p.np+1,0);for(const auto&o:p.obs)if(s.selected[o.point]){s.obs.push_back(o);s.ptr[o.point+1]++;}
    for(int j=0;j<p.np;j++)s.ptr[j+1]+=s.ptr[j];return s;
}
static Mask sample_mask(const Mask&base,const Sample&s){Mask m=base;for(size_t j=0;j<m.point.size();j++)m.point[j]=m.point[j]&&s.selected[j];return m;}
static double objective(const Problem&p,const State&s,const std::vector<Obs>&obs,const std::vector<double>*weights,double*minz=nullptr){
    double f=0.,z=1e100;for(const auto&o:obs){V3 q=s.R[o.cam]*(s.X[o.point]-s.C[o.cam]);z=std::min(z,q[2]);const auto&k=p.K[o.cam];
        double a=k[0]*q[0]/q[2]+k[2]-o.uv[0],b=k[1]*q[1]/q[2]+k[3]-o.uv[1];
        double w=weights?(*weights)[o.point]:1.;f+=w*2.*std::log1p((a*a+b*b)/4.);}
    if(minz)*minz=z;return f;
}
struct System {std::vector<M6> U;std::vector<M3> V;std::vector<M63> W;std::vector<V6> gc;std::vector<V3> gp;double cost=0.;};
static void gradient(const Problem&p,const State&s,const Sample&sample,const Mask&m,System&sys){
    sys.gc.assign(p.nc,V6::Zero());sys.gp.assign(p.np,V3::Zero());sys.cost=0.;std::vector<M3> jl(p.nc);
    for(int k=0;k<p.nc;k++)jl[k]=left_jac(s.O[k]);
    for(const auto&o:sample.obs){int k=o.cam,j=o.point;V3 q=s.R[k]*(s.X[j]-s.C[k]);const auto&K=p.K[k];double iz=1/q[2];
        double r0=K[0]*q[0]*iz+K[2]-o.uv[0],r1=K[1]*q[1]*iz+K[3]-o.uv[1],rr=r0*r0+r1*r1;
        double wt=sample.weights[j],w=wt/(1+rr/4.);sys.cost+=wt*2.*std::log1p(rr/4.);
        V3 gq(w*r0*K[0]*iz,w*r1*K[1]*iz,-w*(r0*K[0]*q[0]+r1*K[1]*q[1])*iz*iz);
        V3 gr=jl[k].transpose()*q.cross(gq),gx=s.R[k].transpose()*gq;
        for(int h=0;h<3;h++){if(m.cam[k][h])sys.gc[k][h]+=gr[h];if(m.cam[k][h+3])sys.gc[k][h+3]-=gx[h];if(m.point[j])sys.gp[j][h]+=gx[h];}
    }
}
static void assemble(const Problem&p,const State&s,const Sample&sample,const Mask&m,System&sys){
    sys.U.assign(p.nc,M6::Zero());sys.V.assign(p.np,M3::Zero());sys.W.assign(sample.obs.size(),M63::Zero());
    sys.gc.assign(p.nc,V6::Zero());sys.gp.assign(p.np,V3::Zero());sys.cost=0.;std::vector<M3>jl(p.nc);
    for(int k=0;k<p.nc;k++)jl[k]=left_jac(s.O[k]);
    for(size_t oi=0;oi<sample.obs.size();oi++){
        const auto&o=sample.obs[oi];int k=o.cam,j=o.point;V3 q=s.R[k]*(s.X[j]-s.C[k]);const auto&K=p.K[k];double iz=1/q[2];
        V2 r(K[0]*q[0]*iz+K[2]-o.uv[0],K[1]*q[1]*iz+K[3]-o.uv[1]);double rr=r.squaredNorm(),wt=sample.weights[j],w=wt/(1+rr/4.);
        sys.cost+=wt*2.*std::log1p(rr/4.);M23 B=M23::Zero();B(0,0)=K[0]*iz;B(1,1)=K[1]*iz;B(0,2)=-K[0]*q[0]*iz*iz;B(1,2)=-K[1]*q[1]*iz*iz;
        M23 jr=(-B*skew(q))*jl[k],jx=B*s.R[k];M26 jc;M23 jp=jx;jc.block<2,3>(0,0)=jr;jc.block<2,3>(0,3)=-jx;
        for(int a=0;a<6;a++)if(!m.cam[k][a])jc.col(a).setZero();if(!m.point[j])jp.setZero();
        for(int a=0;a<6;a++){
            sys.gc[k][a]+=w*(jc(0,a)*r[0]+jc(1,a)*r[1]);
            for(int b=0;b<6;b++)sys.U[k](a,b)+=w*(jc(0,a)*jc(0,b)+jc(1,a)*jc(1,b));
            for(int b=0;b<3;b++)sys.W[oi](a,b)=w*(jc(0,a)*jp(0,b)+jc(1,a)*jp(1,b));}
        for(int a=0;a<3;a++){
            sys.gp[j][a]+=w*(jp(0,a)*r[0]+jp(1,a)*r[1]);for(int b=0;b<3;b++)sys.V[j](a,b)+=w*(jp(0,a)*jp(0,b)+jp(1,a)*jp(1,b));}
    }
    for(int k=0;k<p.nc;k++)for(int a=0;a<6;a++)if(!m.cam[k][a])sys.U[k](a,a)=1.;
    for(int j=0;j<p.np;j++)if(!m.point[j])sys.V[j].setIdentity();
}
struct Factor {
    std::vector<M3> Vi;std::vector<M63> WV;
    Eigen::LLT<Eigen::MatrixXd> llt;
    bool compute(const Problem&p,const Sample&sample,const Mask&m,const System&sys,double lam){
        Eigen::MatrixXd S=Eigen::MatrixXd::Zero(6*p.nc,6*p.nc);Vi.resize(p.np);WV.resize(sample.obs.size());
        for(int k=0;k<p.nc;k++){S.block<6,6>(6*k,6*k)=sys.U[k];for(int a=0;a<6;a++)S(6*k+a,6*k+a)+=lam*m.cam[k][a];}
        for(int j=0;j<p.np;j++){
            double a=sys.V[j](0,0)+lam*m.point[j],b=sys.V[j](0,1),c=sys.V[j](0,2),d=sys.V[j](1,1)+lam*m.point[j],e=sys.V[j](1,2),f=sys.V[j](2,2)+lam*m.point[j];
            double det=a*(d*f-e*e)-b*(b*f-c*e)+c*(b*e-c*d);if(!(det>0.))return false;
            M3&v=Vi[j];v(0,0)=(d*f-e*e)/det;v(0,1)=(c*e-b*f)/det;v(0,2)=(b*e-c*d)/det;
            v(1,0)=v(0,1);v(1,1)=(a*f-c*c)/det;v(1,2)=(b*c-a*e)/det;v(2,0)=v(0,2);v(2,1)=v(1,2);v(2,2)=(a*d-b*b)/det;
            for(int o=sample.ptr[j];o<sample.ptr[j+1];o++)WV[o].noalias()=sys.W[o]*v;
            for(int o=sample.ptr[j];o<sample.ptr[j+1];o++)for(int oo=sample.ptr[j];oo<sample.ptr[j+1];oo++)
                S.block<6,6>(6*sample.obs[o].cam,6*sample.obs[oo].cam).noalias()-=WV[o]*sys.W[oo].transpose();
        }
        S=(.5*(S+S.transpose())).eval();llt.compute(S);return llt.info()==Eigen::Success;
    }
    void solve(const Problem&p,const Sample&sample,const System&sys,std::vector<V6>&dc,std::vector<V3>&dp)const{
        Eigen::VectorXd b(6*p.nc);for(int k=0;k<p.nc;k++)b.segment<6>(6*k)=-sys.gc[k];
        for(int j=0;j<p.np;j++)for(int o=sample.ptr[j];o<sample.ptr[j+1];o++)b.segment<6>(6*sample.obs[o].cam).noalias()+=WV[o]*sys.gp[j];
        Eigen::VectorXd x=llt.solve(b);dc.resize(p.nc);dp.resize(p.np);for(int k=0;k<p.nc;k++)dc[k]=x.segment<6>(6*k);
        for(int j=0;j<p.np;j++){V3 q=-sys.gp[j];for(int o=sample.ptr[j];o<sample.ptr[j+1];o++)q.noalias()-=sys.W[o].transpose()*dc[sample.obs[o].cam];dp[j].noalias()=Vi[j]*q;}
    }
};
struct Options {bool anchors=false,full_only=false;uint32_t seed=11;int hold=4,stage_steps=4,max_iterations=600,min_per_camera=6;double max_seconds=20.;};
struct History {int iteration,sample_points,qbuilds,factors,age;double cost,lambda,elapsed;};
struct Result {
    State state;Plan plan;std::vector<History>history;
    double elapsed=0,assembly=0,linear=0,validation=0,initial=0,final=0;
    int accepted=0,rejected=0,refreshes=0,factors=0,reused=0,rollbacks=0,status=0,last_stage=0;
};
static Result optimize(const Problem&p,const Options&opt){
    auto start=Clock::now();Result out;State s=p.initial;
    if(opt.anchors)for(size_t i=0;i<p.forced.size();i++)s.X[p.forced[i]]=p.anchor_xyz[i];
    Mask base=mask_for(p,s,opt.anchors);out.plan=make_plan(p,opt.seed,opt.full_only,opt.min_per_camera);
    State best=s;double bestcost=objective(p,s,p.obs,nullptr);out.initial=bestcost;
    int stage=0,in_stage=0,age=opt.hold,small=0;double lam=-1.,next_lam=-1.;
    Sample sample=make_sample(p,out.plan,out.plan.sizes[stage]);Mask m=sample_mask(base,sample);System sys;Factor fact;
    for(int it=0;it<opt.max_iterations;it++){
        if(seconds(start)>opt.max_seconds){out.status=1;break;}
        bool fresh=age>=opt.hold;
        if(fresh){
            auto t=Clock::now();assemble(p,s,sample,m,sys);out.assembly+=seconds(t);out.refreshes++;
            if(lam<0){double mx=0;for(const auto&u:sys.U)mx=std::max(mx,u.maxCoeff());lam=1e-6*std::max(1.,mx);}else if(next_lam>0)lam=next_lam;
            t=Clock::now();if(!fact.compute(p,sample,m,sys,std::max(lam,1e-12)))throw std::runtime_error("Initial factorization failed");out.linear+=seconds(t);out.factors++;age=0;next_lam=lam;
        }else{auto t=Clock::now();gradient(p,s,sample,m,sys);out.assembly+=seconds(t);}
        bool ok=false,was_reused=age>0;double fn=sys.cost,fc=sys.cost;
        for(int trial=0;trial<20;trial++){
            std::vector<V6>dc;std::vector<V3>dp;auto t=Clock::now();fact.solve(p,sample,sys,dc,dp);out.linear+=seconds(t);
            State sn=s;double maxangle=0;
            for(int k=0;k<p.nc;k++){sn.O[k]+=dc[k].head<3>();sn.C[k]+=dc[k].tail<3>();sn.R[k]=exp3(sn.O[k])*p.initial.R[k];maxangle=std::max(maxangle,dc[k].head<3>().squaredNorm());}
            for(int j=0;j<p.np;j++)sn.X[j]+=dp[j];double minz;fn=objective(p,sn,sample.obs,&sample.weights,&minz);
            if(fn<fc&&minz>1e-8&&maxangle<.35*.35){
                double rel=(fc-fn)/std::max(1.,fc);s=std::move(sn);age++;in_stage++;out.accepted++;if(was_reused)out.reused++;
                next_lam=std::max(next_lam/3,1e-12);out.history.push_back({it,out.plan.sizes[stage],out.refreshes,out.factors,age,fn,lam,seconds(start)});
                small=(stage==int(out.plan.sizes.size())-1&&rel<1e-10)?small+1:0;ok=true;break;
            }
            out.rejected++;
            if(age>0){t=Clock::now();assemble(p,s,sample,m,sys);out.assembly+=seconds(t);out.refreshes++;age=0;was_reused=false;fc=sys.cost;}
            lam=std::max(1e-8,lam*4);next_lam=lam;t=Clock::now();
            if(!fact.compute(p,sample,m,sys,lam))throw std::runtime_error("Damped factorization failed");out.linear+=seconds(t);out.factors++;
        }
        if(!ok){if(stage<int(out.plan.sizes.size())-1)in_stage=opt.stage_steps;else{out.status=3;break;}}
        bool transition=stage<int(out.plan.sizes.size())-1&&in_stage>=opt.stage_steps;
        if(transition||stage==int(out.plan.sizes.size())-1){
            double fullcost;
            if(stage==int(out.plan.sizes.size())-1)fullcost=ok?fn:fc;else{auto t=Clock::now();fullcost=objective(p,s,p.obs,nullptr);out.validation+=seconds(t);}
            if(fullcost<bestcost){bestcost=fullcost;best=s;}
            else if(transition&&fullcost>bestcost*1.001){s=best;out.rollbacks++;}
            if(transition){stage++;in_stage=0;age=opt.hold;small=0;lam=-1.;next_lam=-1.;sample=make_sample(p,out.plan,out.plan.sizes[stage]);m=sample_mask(base,sample);}
        }
        if(stage==int(out.plan.sizes.size())-1&&small>=3){out.status=4;break;}
    }
    auto t=Clock::now();double finalcurrent=objective(p,s,p.obs,nullptr);out.validation+=seconds(t);
    if(finalcurrent<bestcost){bestcost=finalcurrent;best=s;}
    out.state=std::move(best);out.final=bestcost;out.last_stage=stage;out.elapsed=seconds(start);return out;
}
struct Metrics {double rotation_mean_deg=0,center_rmse_mm=0,heldout_rmse_px=0;};
static Metrics metrics(const Problem&p,const State&s){
    V3 ma=V3::Zero(),mb=V3::Zero();for(int k=0;k<p.nc;k++){ma+=s.C[k];mb+=p.Cgt[k];}ma/=p.nc;mb/=p.nc;
    M3 cov=M3::Zero();double var=0;for(int k=0;k<p.nc;k++){V3 a=s.C[k]-ma,b=p.Cgt[k]-mb;cov.noalias()+=b*a.transpose()/p.nc;var+=a.squaredNorm()/p.nc;}
    Eigen::JacobiSVD<M3>svd(cov,Eigen::ComputeFullU|Eigen::ComputeFullV);V3 signs(1,1,(svd.matrixU()*svd.matrixV().transpose()).determinant());
    M3 Q=svd.matrixU()*signs.asDiagonal()*svd.matrixV().transpose();double scale=svd.singularValues().dot(signs)/var;V3 trans=mb-scale*Q*ma;
    Metrics m;double ce=0,pe=0;for(int k=0;k<p.nc;k++){
        M3 ra=s.R[k]*Q.transpose();double tr=(ra*p.Rgt[k].transpose()).trace();m.rotation_mean_deg+=std::acos(std::clamp((tr-1)/2,-1.,1.))*180/std::acos(-1.)/p.nc;
        ce+=(scale*Q*s.C[k]+trans-p.Cgt[k]).squaredNorm();}
    m.center_rmse_mm=std::sqrt(ce/p.nc)*p.unit*1000;
    for(const auto&o:p.test){V3 q=s.R[o.cam]*(s.X[o.point]-s.C[o.cam]);const auto&k=p.K[o.cam];V2 e(k[0]*q[0]/q[2]+k[2]-o.uv[0],k[1]*q[1]/q[2]+k[3]-o.uv[1]);pe+=e.squaredNorm();}
    m.heldout_rmse_px=std::sqrt(pe/p.test.size());return m;
}
static void self_test(const Problem&p,const Options&opt){
    State s=p.initial;if(opt.anchors)for(size_t i=0;i<p.forced.size();i++)s.X[p.forced[i]]=p.anchor_xyz[i];
    Mask base=mask_for(p,s,opt.anchors);Plan plan=make_plan(p,opt.seed,false,6);Sample sample=make_sample(p,plan,plan.sizes[0]);Mask m=sample_mask(base,sample);
    for(int k=0;k<p.nc;k++){s.O[k]=V3(.003*std::sin(k+1),.002*std::cos(k+1),.001*std::sin(2*k+1));s.R[k]=exp3(s.O[k])*p.initial.R[k];}
    System a,b;assemble(p,s,sample,m,a);gradient(p,s,sample,m,b);double ge=0,gs=1;
    for(int k=0;k<p.nc;k++){ge=std::max(ge,(a.gc[k]-b.gc[k]).cwiseAbs().maxCoeff());gs=std::max(gs,a.gc[k].cwiseAbs().maxCoeff());}
    for(int j=0;j<p.np;j++){ge=std::max(ge,(a.gp[j]-b.gp[j]).cwiseAbs().maxCoeff());gs=std::max(gs,a.gp[j].cwiseAbs().maxCoeff());}
    if(ge/gs>1e-11||std::abs(a.cost-b.cost)>1e-8)throw std::runtime_error("Gradient-only/assembly mismatch");
    std::mt19937 rng(901);std::normal_distribution<double>normal;double fdmax=0;
    for(int trial=0;trial<5;trial++){
        std::vector<V6>dc(p.nc);std::vector<V3>dp(p.np);double sz=0;
        for(int k=0;k<p.nc;k++)for(int h=0;h<6;h++){dc[k][h]=m.cam[k][h]?normal(rng):0.;sz+=dc[k][h]*dc[k][h];}
        for(int j=0;j<p.np;j++)for(int h=0;h<3;h++){dp[j][h]=m.point[j]?normal(rng):0.;sz+=dp[j][h]*dp[j][h];}
        sz=std::sqrt(sz);for(auto&v:dc)v/=sz;for(auto&v:dp)v/=sz;
        auto f=[&](double h){State sn=s;for(int k=0;k<p.nc;k++){sn.O[k]+=h*dc[k].head<3>();sn.C[k]+=h*dc[k].tail<3>();sn.R[k]=exp3(sn.O[k])*p.initial.R[k];}for(int j=0;j<p.np;j++)sn.X[j]+=h*dp[j];return objective(p,sn,sample.obs,&sample.weights);};
        double fd=(f(1e-6)-f(-1e-6))/(2e-6),exact=0;for(int k=0;k<p.nc;k++)exact+=a.gc[k].dot(dc[k]);for(int j=0;j<p.np;j++)exact+=a.gp[j].dot(dp[j]);
        fdmax=std::max(fdmax,std::abs(fd-exact)/std::max(1.,std::abs(fd)));
    }
    if(fdmax>1e-7)throw std::runtime_error("Finite-difference gradient check failed");
    double mx=0;for(auto&u:a.U)mx=std::max(mx,u.maxCoeff());double lam=1e-6*std::max(1.,mx);Factor factor;
    if(!factor.compute(p,sample,m,a,lam))throw std::runtime_error("Test factorization failed");
    std::vector<V6>dc;std::vector<V3>dp;factor.solve(p,sample,a,dc,dp);std::vector<V6>rc(p.nc);std::vector<V3>rp(p.np);
    double normg=0;for(int k=0;k<p.nc;k++){rc[k]=a.U[k]*dc[k]+a.gc[k];for(int h=0;h<6;h++)rc[k][h]+=lam*m.cam[k][h]*dc[k][h];normg+=a.gc[k].squaredNorm();}
    for(int j=0;j<p.np;j++){rp[j]=a.V[j]*dp[j]+a.gp[j]+lam*m.point[j]*dp[j];normg+=a.gp[j].squaredNorm();}
    for(size_t o=0;o<sample.obs.size();o++){int k=sample.obs[o].cam,j=sample.obs[o].point;rc[k].noalias()+=a.W[o]*dp[j];rp[j].noalias()+=a.W[o].transpose()*dc[k];}
    double normr=0;for(auto&v:rc)normr+=v.squaredNorm();for(auto&v:rp)normr+=v.squaredNorm();double lr=std::sqrt(normr/std::max(1.,normg));
    if(lr>1e-8)throw std::runtime_error("Schur solve residual check failed");
    std::cout<<std::setprecision(12)<<"SELF_TEST_PASSED gradient_relative_error="<<ge/gs<<" finite_difference_relative_error="<<fdmax<<" linear_system_relative_residual="<<lr<<'\n';
}
static void save_state(const Problem&p,const State&s,const std::string&path){
    std::ofstream f(path,std::ios::binary);if(!f)throw std::runtime_error("Cannot write state");f.write("SDCAOUT1",8);uint32_t h[2]={uint32_t(p.nc),uint32_t(p.np)};writev(f,h,2);
    for(const auto&R:s.R)for(int a=0;a<3;a++)for(int b=0;b<3;b++)writev(f,&R(a,b),1);
    for(const auto&v:s.C)writev(f,v.data(),3);for(const auto&v:s.X)writev(f,v.data(),3);
}
static void save_plan(const Plan&p,const std::string&path){
    std::ofstream f(path,std::ios::binary);uint32_t n=uint32_t(std::count(p.core.begin(),p.core.end(),uint8_t(1)));writev(f,&n,1);
    for(size_t j=0;j<p.core.size();j++)if(p.core[j]){uint32_t q=uint32_t(j);writev(f,&q,1);}n=uint32_t(p.perm.size());writev(f,&n,1);for(int j:p.perm){uint32_t q=uint32_t(j);writev(f,&q,1);}
}
static void save_json(const Problem&p,const Options&opt,const Result&r,const Metrics&m,const std::string&path,double warmup){
    std::ofstream f(path);if(!f)throw std::runtime_error("Cannot write JSON");f<<std::setprecision(17);
    const char*statuses[]={"max_iterations","time_limit","unused","no_decreasing_step","small_full_cost_change"};
    f<<"{\n\"algorithm\":\"CPP_progressive_sampled_local_DCA\",\n\"seconds\":"<<r.elapsed
     <<",\n\"assembly_seconds\":"<<r.assembly<<",\n\"linear_solver_seconds\":"<<r.linear
     <<",\n\"assembly_and_control_seconds\":"<<r.elapsed-r.linear<<",\n\"validation_seconds\":"<<r.validation
     <<",\n\"initial_cost\":"<<r.initial<<",\n\"final_cost\":"<<r.final<<",\n\"accepted\":"<<r.accepted
     <<",\n\"rejected\":"<<r.rejected<<",\n\"Q_rebuilds\":"<<r.refreshes<<",\n\"factorizations\":"<<r.factors
     <<",\n\"steps_reusing_factorization\":"<<r.reused<<",\n\"global_checkpoint_rollbacks\":"<<r.rollbacks
     <<",\n\"status\":\""<<statuses[r.status]<<"\",\n\"mandatory_point_count\":"<<std::count(r.plan.core.begin(),r.plan.core.end(),uint8_t(1))
     <<",\n\"minimum_camera_core_coverage\":"<<*std::min_element(r.plan.counts.begin(),r.plan.counts.end())
     <<",\n\"sample_seed\":"<<opt.seed<<",\n\"Q_hold_accepted_steps\":"<<opt.hold
     <<",\n\"full_only_ablation\":"<<(opt.full_only?"true":"false")<<",\n\"H_convexity_certified\":false,\n\"warmup_seconds_excluded\":"<<warmup
     <<",\n\"sample_sizes\":[";
    for(size_t i=0;i<r.plan.sizes.size();i++){if(i)f<<',';f<<r.plan.sizes[i];}
    f<<"],\n\"metrics\":{\"rotation_mean_deg\":"<<m.rotation_mean_deg<<",\"camera_center_rmse_mm_sim3\":"<<m.center_rmse_mm<<",\"heldout_reprojection_rmse_px\":"<<m.heldout_rmse_px<<"},\n\"history\":[";
    for(size_t i=0;i<r.history.size();i++){if(i)f<<',';auto&h=r.history[i];f<<"{\"iteration\":"<<h.iteration<<",\"batch_cost\":"<<h.cost<<",\"sample_points\":"<<h.sample_points<<",\"Q_rebuilds\":"<<h.qbuilds<<",\"factorizations\":"<<h.factors<<",\"Q_age\":"<<h.age<<",\"lambda\":"<<h.lambda<<",\"elapsed_seconds\":"<<h.elapsed<<'}';}
    f<<"]\n}\n";
}
int main(int argc,char**argv){
    try{
        Eigen::setNbThreads(1);Options opt;std::string input,output="result.json",statepath,planpath;bool warm=true,test=false;
        for(int i=1;i<argc;i++){
            std::string a=argv[i];auto value=[&](){if(i+1>=argc)throw std::runtime_error("Missing argument");return std::string(argv[++i]);};
            if(a=="--input")input=value();else if(a=="--output")output=value();else if(a=="--state")statepath=value();else if(a=="--plan")planpath=value();
            else if(a=="--anchors")opt.anchors=std::stoi(value())!=0;else if(a=="--full-reuse")opt.full_only=true;
            else if(a=="--seed")opt.seed=uint32_t(std::stoul(value()));else if(a=="--max-iterations")opt.max_iterations=std::stoi(value());
            else if(a=="--hold")opt.hold=std::stoi(value());else if(a=="--max-seconds")opt.max_seconds=std::stod(value());else if(a=="--no-warmup")warm=false;
            else if(a=="--self-test")test=true;
            else throw std::runtime_error("Unknown option: "+a);
        }
        if(input.empty())throw std::runtime_error("Usage: stochastic_dca.exe --input problem.bin --output result.json [--anchors 0|1] [--seed 11] [--full-reuse]");
        Problem p=load(input);if(test){self_test(p,opt);return 0;}double warmtime=0;if(warm){Options w=opt;w.max_iterations=1;auto t=Clock::now();optimize(p,w);warmtime=seconds(t);}
        Result result=optimize(p,opt);Metrics met=metrics(p,result.state);save_json(p,opt,result,met,output,warmtime);
        if(!statepath.empty())save_state(p,result.state,statepath);if(!planpath.empty())save_plan(result.plan,planpath);
        std::cout<<std::setprecision(10)<<"seconds="<<result.elapsed<<" cost="<<result.final<<" steps="<<result.accepted<<" Q="<<result.refreshes<<" camera_rmse_mm="<<met.center_rmse_mm<<'\n';return 0;
    }catch(const std::exception&e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 1;}
}
