"""An expandable, source-linked formula notebook, not a claim of exhaustiveness."""

# Repository layout adapter; numerical methods below retain their prior equations.
import sys as _sys
from pathlib import Path as _Path
from functools import partial as _partial
_PROJECT_ROOT = next(p for p in _Path(__file__).resolve().parents if (p / "project.json").is_file())
_sys.path.insert(0, str(_PROJECT_ROOT))
from project_paths import PROJECT_ROOT, RAW_WAV, LEO_SNAPSHOT, EXTERNAL_LEO_ROOT, VI_ROOT, legacy_artifact, experiment_output, require_run_context
artifact_path = _partial(legacy_artifact, prefix='learning', experiment='LEARNING')
if __name__ == "__main__":
    require_run_context()
from pathlib import Path
import json

OUT=experiment_output('LEARNING', 'learning');OUT.mkdir(exist_ok=True)
sources=[
('G86','Gardner, The Spectral Correlation Theory of Cyclostationary Time-Series (1986)','https://faculty.engineering.ucdavis.edu/gardner/wp-content/uploads/sites/146/2013/02/The_Spectral_Correlation_Theory_of_Cyclostationary_Time-Series.pdf','作者所在大学公开原论文，全文可读'),
('GMEAS','Gardner, Measurement of Spectral Correlation (1986)','https://cyclostationarity.com/wp-content/uploads/2022/10/202210171025.pdf','作者研究资料站原论文，检索摘要可读，PDF正文提取受限'),
('DG94','Dandawate and Giannakis, Statistical Tests for Presence of Cyclostationarity (1994)','https://experts.umn.edu/en/publications/statistical-tests-for-presence-of-cyclostationarity/','作者大学页面与摘要；未声称复现其全部渐近检验'),
('SPY','SciPy Signal Processing official guide','https://docs.scipy.org/doc/scipy/tutorial/signal.html','官方实现与定义；不把软件文档当作信号体制证据'),
('ZOOM','SciPy zoom_fft official reference','https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.zoom_fft.html','官方频率区间DFT接口，已用于频率细化'),
('DELAY','SciPy group_delay official reference','https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.group_delay.html','官方群时延定义'),
('CONV','MIT 6.02 Chapter 7: Convolutional Codes','https://ocw.mit.edu/courses/6-02-introduction-to-eecs-ii-digital-communication-systems-fall-2012/6bb326417ac947837cce79d334c8ac1c_MIT6_02F12_chap07.pdf','MIT原课程章节，全文可读'),
('VIT','MIT 6.02 Chapter 8: Viterbi Decoding','https://ocw.mit.edu/courses/6-02-introduction-to-eecs-ii-digital-communication-systems-fall-2012/effb4e0089fd355258e96dd8277bbc2b_MIT6_02F12_chap08.pdf','MIT原课程章节，全文可读'),
('LDPC','Gallager, Low Density Parity Check Codes, MIT thesis (1960)','https://dspace.mit.edu/entities/publication/6efcd810-4862-40aa-b4e7-88dc677a413c','MIT原学位论文目录；未在此重新分发PDF'),
('POLAR','Arikan, Channel Polarization (2008/2009)','https://arxiv.org/abs/0807.3917','原作者预印本v5元数据与摘要已核对'),
('CRC','Philip Koopman, CRC research tables and notation','https://users.ece.cmu.edu/~koopman/crc/','作者大学研究资源，记号与适用假设需核对'),
('CCSDS','CCSDS 131.0-B-6, TM Synchronization and Channel Coding (April 2026)','https://ccsds.org/wp-content/uploads/gravity_forms/5-448e85c647331d9cbaf66c096458bdd5/2026/06/131x0b6ec1.pdf','官方现行条目与PDF已核对；不是Starlink标准'),
('DVBS2','ETSI EN 302 307-1 V1.4.1, DVB-S2 (2014-11)','https://www.etsi.org/deliver/etsi_en/302300_302399/30230701/01.04.01_60/en_30230701v010401p.pdf','官方定稿历史版本，作为编码调制参考，不声称最新全部DVB系列或Starlink采用它'),
('HUM','Humphreys et al., Signal Structure of the Starlink Ku-Band Downlink','https://arxiv.org/abs/2210.11578','原研究v3 (2023)，结构与同步参考'),
('QIN','Qin et al., Pilots and Other Predictable Elements of the Starlink Ku-Band Downlink','https://arxiv.org/html/2602.02627v1','本地模板对应的原预印本v1；已有2026期刊版，不混用版本参数'),
('LIU','Liu et al., Protocol Data Unit Header Decoding and Payload Coding Recovery in the Starlink Ku-Band Downlink','https://arxiv.org/html/2610.29387v2','原研究v2 (2026-10-29)，布局中的条件解释保持原有边界'),
('MIT','Gallager, MIT 6.450 Principles of Digital Communications I','https://ocw.mit.edu/courses/6-450-principles-of-digital-communications-i-fall-2006/','原课程总入口及公开章节；通用公式以本手册统一记号重述'),
('SHANNON','Shannon, A Mathematical Theory of Communication (1948)','https://www.nokia.com/bell-labs/publications-and-media/publications/a-mathematical-theory-of-communication/','Bell Labs原出版机构档案'),
('BEEK','van de Beek, Sandell and Borjesson, ML Estimation of Time and Frequency Offset in OFDM Systems (1997)','https://ltu.diva-portal.org/smash/get/diva2%3A989074/FULLTEXT01','原作者大学仓库PDF检索内容已核对；本轮直接全文获取超时'),
('TED','Gardner, A BPSK/QPSK Timing-Error Detector for Sampled Receivers (1986)','https://doi.org/10.1109/TCOM.1986.1096561','原论文DOI元数据已核对；全文访问受限，公式为通用算法记号而非逐式转录')]
S={r[0]:{'id':r[0],'title':r[1],'url':r[2],'verification':r[3],'checked_on':'2026-11-20'} for r in sources}
cards=[]
def add(id,group,title,latex,meaning,conditions,use,refs):
    cards.append({'id':id,'group':group,'title':title,'latex':latex,'meaning':meaning,'conditions':conditions,'local_application':use,'sources':refs})

add('F01','观测与预处理','复基带观测模型',r'y[n]=\sum_p(h_p*s_p)(nT_s/(1+\epsilon)-\tau_p)e^{j\phi_p[n]}+w[n]',
    '把多源、通道、时间尺度误差和相位演化分开记账。接收LO可进入多个分量的共同相位项。','是建模框架；未知参数之间可能不可辨识。','用于组织整段WAV；不把多分量默认合并为同一发射源。',['MIT','QIN'])
add('F02','观测与预处理','频率与相位及去频漂',r'f(t)=\frac{1}{2\pi}\frac{d\phi}{dt},\quad z(t)=y(t)e^{-j\phi(t)};\quad \phi(t)=2\pi(c_2t^3/3+c_1t^2/2+c_0t)',
    '二次频率轨迹需要积分成三次相位再补偿。','频率系数须有单位；减相位不等于重采样校正。','已用于细线和宽谱关联检验；轨迹来自较早整段测量。',['SPY','QIN'])
add('F03','观测与预处理','接收本振与采样钟误差',r'f_{\rm meas}=\frac{f_{\rm RF}-f_{\rm LO}}{1+\epsilon}\simeq f_{\rm RF}-f_{\rm LO}-\epsilon(f_{\rm RF}-f_{\rm LO})',
    'LO误差是加性频偏，采样率误差还有频率比例项。','采用实际采样率等于名义值乘(1+epsilon)的约定。','单份录波不能唯一拆分传播多普勒、发射钟和接收钟。',['MIT','QIN'])
add('F04','观测与预处理','I/Q镜像的广义线性模型',r'y(t)=a\,x(t)+b\,x^*(t);\quad \kappa=\frac{|E[y^2]|}{E[|y|^2]}',
    '共轭泄漏可产生镜像；圆度指标测量非圆性。','先去均值；非零指标也可能来自真实非圆调制，不能唯一归因硬件。','五轴相位检查未支持强稳定共轭模型，不等于排除所有镜像。',['MIT','G86'])
add('F05','观测与预处理','短时傅里叶变换',r'X_m[k]=\sum_{n=0}^{N-1}x[n+mH]w[n]e^{-j2\pi kn/N}',
    '把时间变化与频率结构联合表示。','FFT频点间隔Fs/N不等于真实可分辨线宽；窗和观察长度都重要。','瀑布与SCF使用不同窗长，各自报告分辨率。',['SPY'])
add('F06','观测与预处理','Welch功率谱与频带功率',r'\hat P[k]=\frac{1}{MF_s\sum_n|w[n]|^2}\sum_m|X_m[k]|^2,\quad P_B\simeq\sum_{k\in B}\hat P[k]\Delta f',
    '用正确归一化把谱密度积分为功率。','复IQ使用双边谱；未校准只能报告数字功率，不称dBm。','用事件前后谱差分离变化成分，但事件后并非纯噪声。',['SPY'])
add('F07','观测与预处理','群时延与边沿展宽',r'\tau_g(\omega)=-\frac{d\arg H(e^{j\omega})}{d\omega},\quad D_{\rm FIR}=(L-1)/2',
    '对称线性相位FIR的因果延迟可扣除，响应宽度仍然保留。','D以样本计；居中离线卷积不能消除真实接收前端未知时延。','终止沿已采用滤波响应模型，未宣称微秒精度。',['DELAY'])
add('F08','观测与预处理','门控细线与宽带功率边沿',r'P_{\rm CW}(t)\propto|h*u_{\rm off}|^2,\quad P_{\rm white}(t)\propto\int_{\xi>t-t_0}|h(\xi)|^2d\xi',
    '相干门控纯音与不相干宽带噪声经过同一滤波器也有不同功率响应。','前式须局部相干，后式须局部白化；复杂调制可能不满足。','用于终止沿拟合与带宽敏感性检查。',['SPY','DELAY'])
add('F09','循环统计','普通二阶循环自相关 CAF',r'R_x^\alpha(\tau)=\lim_{T\to\infty}\frac1T\int_{-T/2}^{T/2}x(t+\tau/2)x^*(t-\tau/2)e^{-j2\pi\alpha t}dt',
    '寻找相关结构随时间的周期变化；alpha是循环频率，tau是相关时延。','有限记录需要窗、噪声对照和多重搜索校正。','本轮先扫描50..6000Hz及11个时延，再固定候选到留出块。',['G86','GMEAS'])
add('F10','循环统计','谱相关函数 SCF',r'S_x^\alpha(f)=\int R_x^\alpha(\tau)e^{-j2\pi f\tau}d\tau\;\longleftrightarrow\;\langle X(f+\alpha/2)X^*(f-\alpha/2)\rangle',
    '把CAF在时延方向变换，显示循环结构分布在哪些频率。','频域估计需一致的时间中心、相位及平滑约定。','已输出三个子带的频率分辨SCF；二阶SCF不是高阶累积量。',['G86','GMEAS'])
add('F11','循环统计','循环谱相干归一化',r'\gamma_x^\alpha(f)=\frac{S_x^\alpha(f)}{\sqrt{S_x^0(f+\alpha/2)S_x^0(f-\alpha/2)}}',
    '减少频带功率差对谱相关比较的影响。','有限样本幅度有正偏；不能仅凭图中最大值宣布检出。','SCF剖面用于诊断，显著性另由冻结候选与联合对照给出。',['G86'])
add('F12','循环统计','共轭循环相关与普通相关的区别',r'C_x^\alpha(\tau)=\langle x(t+\tau/2)x(t-\tau/2)e^{-j2\pi\alpha t}\rangle',
    '不取第二项共轭的统计可见非圆性，和普通SCF携带不同信息。','不要把普通SCF中的复共轭记号与“共轭循环统计”混称。','本轮新SCF是普通二阶；此前共轭边带检查是另一项实验。',['G86'])
add('F13','循环统计','相位演化对相关乘积的精确作用',r'x=s e^{j\phi}\Rightarrow x(t+\tau/2)x^*(t-\tau/2)=s(t+\tau/2)s^*(t-\tau/2)e^{j[\phi(t+\tau/2)-\phi(t-\tau/2)]}',
    '把频漂对循环频率的影响从相关定义直接推导出来。','恒等式成立；把它近似成单一频移还需要局部平滑。','这是本轮解释时延相关频移的关键，本手册推导而非声称某篇论文原式。',['G86'])
add('F14','循环统计','循环频率的时延相关漂移',r'\alpha_{\rm obs}(\tau,t)\simeq\alpha_0+\tau\dot f(t)',
    '线性频漂产生与相关时延成比例的循环频率偏移。','单位为秒乘Hz/秒；非线性频漂使峰展宽，不能误认成不同符号率。','本地raw候选偏移与细线轨迹预测最大差约0.042Hz；去漂后宽谱10候选落到750Hz整数倍的搜索网格。',['G86'])
add('F15','循环统计','冻结候选与联合随机相位检验',r'T_j=\frac{|\sum_b c_{bj}|^2}{\sum_b|c_{bj}|^2},\quad p_j=\frac{1+\#\{r:\max_qT_q^{(r)}\ge T_j\}}{R+1}',
    '对所有冻结候选构造同一最大统计量对照族，控制这一族的多重搜索。','本项目采用的重采样检验，不等同DG94原文渐近检验；依赖整块相位对称假设。','每个坐标18项候选含15项盲选和3项历史750Hz先验；最后合并为36项联合检验。999次对照的最小经验p为0.001。',['DG94','GMEAS'])
add('F16','同步与调制','匹配滤波/已知序列相关',r'c[\ell]=\sum_n y[n]s^*[n-\ell]',
    '已知波形的相干积累可提高检测灵敏度。','须核对频偏、采样率、带限和模板；训练寻峰与留出评价分开。','已知PSS模板此前未显示足够特异性，不因新循环结构而自动升级为解码成功。',['MIT','HUM'])
add('F17','同步与调制','重复段定时与频偏',r'P(d)=\sum_{n=0}^{L-1}r[d+n+L]r^*[d+n],\quad \hat f=\frac{F_s}{2\pi L}\arg P(d)',
    '利用已确认的重复段提取定时和分数频偏。','重复训练或其他重复结构必须真实存在；频偏估计有模Fs/L歧义。','列为条件方法，不把任意相关峰当作Starlink训练段。',['BEEK','MIT'])
add('F18','同步与调制','OFDM循环前缀相关',r'C(d)=\sum_{n=0}^{N_{CP}-1}r[d+n+N]r^*[d+n]',
    'CP复制关系可约束有效符号长度与频偏。','必须有足够带宽与时间分辨能力；窄带投影下原CP关系不能原样照搬。','本WAV为窄带截取，无法把当前750Hz周期直接解释成CP检测。',['BEEK','HUM'])
add('F19','同步与调制','Gardner定时误差检测',r'e_k=\Re\{(z_{k-1}-z_k)z^*_{k-1/2}\}',
    '用早、晚和半符号采样驱动定时环。','正负号依环路约定；需要合适的匹配滤波、采样与调制条件。','尚无可靠单载波符号模型，暂不盲套定时环。',['TED','MIT'])
add('F20','同步与调制','M-PSK幂次相位估计',r'z_k^M\approx A^M e^{jM\phi},\quad\hat\phi=\frac1M\arg\sum_k z_k^M',
    '去掉理想MPSK符号的离散相位，估计公共载波相位。','存在2pi/M歧义；QAM、脉冲过渡、OFDM叠加不满足简单模型。','作为候选体制下的方法，不对整个宽谱直接四次方判成QPSK。',['MIT'])
add('F21','同步与调制','线性调制与Nyquist脉冲',r's(t)=\sum_k a_kp(t-kT),\quad p(mT)=\delta[m]',
    '脉冲与符号率共同决定频谱和循环特征。','Nyquist零ISI条件用于整体脉冲响应；实际通道和接收滤波要计入。','循环频率不一定就是符号率，还可能是帧、门控或导频周期。',['MIT'])
add('F22','同步与调制','PSK和QAM候选星座',r'a_m=Ae^{j2\pi m/M};\quad a_{uv}=d[(2u-L+1)+j(2v-L+1)]',
    '前者为MPSK，后者为M=L平方的方形QAM示意。','幅度归一化、Gray映射、相位和位序均需确认；并非所有QAM都是方形。','目前未获得宽谱单符号星座，不能凭整体I/Q散点直接命名。',['MIT','DVBS2'])
add('F23','同步与调制','OFDM合成与解调',r'x_m[n]=\frac1{\sqrt N}\sum_{k=0}^{N-1}X_m[k]e^{j2\pi kn/N},\quad\hat X_m[k]=\frac1{\sqrt N}\sum_nx_m[n]e^{-j2\pi kn/N}',
    '用正交子载波承载一组调制符号，CP另行添加/去除。','有限采集带宽只保留部分子载波；不能从缺失频带补造全帧。','NPZ导频片段可验证局部结构，原WAV仍需独立证据。',['HUM','BEEK'])
add('F24','同步与调制','OFDM频率与时间网格',r'\Delta f=1/T_u,\quad T_{sym}=T_u+T_{CP},\quad F_s=N/T_u',
    '区分子载波间隔、OFDM槽间隔与帧周期。','公式Fs是完整处理网格，不一定等于窄带录波采样率。','公开Starlink网格与本地750Hz循环证据分别记录，不能互相替代证明。',['HUM'])
add('F25','信道与软信息','导频LS和线性MMSE信道估计',r'\hat h_{LS}=(A^HA)^{-1}A^Hy;\quad\hat h_{MMSE}=R_hA^H(AR_hA^H+R_n)^{-1}y',
    '在已知设计矩阵下估计通道；有先验协方差时可作MMSE。','LS需满列秩；病态、插值误差和未知符号泄漏要检查。','已知导频与未知子空间分离的投影提取器是本地实现分支。',['MIT','QIN'])
add('F26','信道与软信息','单抽头MMSE均衡',r'\hat a_k=\frac{H_k^*y_k}{|H_k|^2+\sigma_n^2/\sigma_a^2}',
    '在噪声与通道增益之间折中，避免深衰落时ZF过度放大噪声。','需要可信H和噪声/符号方差；残余频偏引起ICI时单抽头可能失效。','未把导频外的相位外推当作已验证绝对比特判决。',['MIT','QIN'])
add('F27','信道与软信息','比特软信息LLR',r'L_i=\log\frac{\sum_{a:b_i=0}\exp(-|y-Ha|^2/N_0)}{\sum_{a:b_i=1}\exp(-|y-Ha|^2/N_0)}',
    '保留判决置信度，供软译码使用。','此式为等先验、复AWGN和已知映射；噪声定义与接收机须一致。','未知星座、位序或绝对相位时不能把相位候选直接当LLR。',['MIT','VIT'])
add('F28','信道与软信息','EVM与差分观测',r'EVM=\sqrt{\frac{\sum_k|z_k-a_k|^2}{\sum_k|a_k|^2}},\quad d_k=z_kz_{k-1}^*',
    'EVM衡量参考符号误差；差分可消去恒定初相。','参考符号必须已知或独立判定；差分不自动移除频偏且会改变噪声。','已知导频一致度不等于未知载荷BER，差分候选仍需校验。',['QIN','MIT'])
add('F29','纠错与校验','生成矩阵、校验矩阵与综合',r'c=uG\pmod2,\quad Hc^T=0,\quad s=H\hat c^T',
    '线性码把信息向量映射为满足校验约束的码字。','需知道分块、穿孔、缩短和比特序；零综合不解释应用语义。','目前本地尚无完整头部码字供消息级校验。',['LDPC','CCSDS'])
add('F30','纠错与校验','卷积编码',r'c_a[n]=\bigoplus_{\ell=0}^{K-1}g_{a,\ell}u[n-\ell]',
    '移位寄存器抽头生成各路编码输出。','抽头方向、输出排列、初态、尾位和极性须完整声明。','公开Starlink短块采用K=7、三路输出；本地头部覆盖不足。',['CONV','LIU'])
add('F31','纠错与校验','Viterbi状态递推',r"M_t(s)=\min_{s'\to s}\{M_{t-1}(s')+\lambda_t(s',s)\}",
    '保留每个状态的最优路径，避免穷举全部输入串。','分支代价须匹配硬判或LLR噪声模型；最好以重编码和CRC复核。','这是有编码位以后才进入的分支，不对原始IQ直接运行Viterbi。',['VIT'])
add('F32','纠错与校验','CRC多项式余数',r'r(x)=x^mM(x)\bmod g(x),\quad C(x)=x^mM(x)+r(x)',
    '展示零初值、无最终异或的系统式CRC基本关系。','实际init/refin/refout/xorout与位序会产生仿射变化；不能只写多项式就实现完整CRC。','本地研究参考头部为mask0x4D、init0xFF，仍未得到可验证完整消息。',['CRC','LIU'])
add('F33','纠错与校验','LFSR与加扰/解扰',r'p[n]=\bigoplus_i a_i p[n-i],\quad b_{tx}[n]=b[n]\oplus p[n]',
    '线性伪随机序列可用于加扰、参考相位或同步。','初始状态、抽头、复位点及串并序必须确认；加扰不是加密。','两种公开相位模板可相差二值周期模式，不能把该差异当消息。',['CCSDS','QIN','LIU'])
add('F34','纠错与校验','LDPC置信传播',r"L_{v\to c}=L_v+\sum_{c'\ne c}L_{c'\to v},\quad L_{c\to v}=2\operatorname{atanh}\prod_{v'\ne v}\tanh(L_{v'\to c}/2)",
    '变量节点汇总软信息，校验节点施加奇偶约束。','求和/乘积限于图邻居；实现需数值截断、调度和停止准则。','只有完整或明确擦除模式的码字、可信校验矩阵和LLR才可使用。',['LDPC','CCSDS'])
add('F35','纠错与校验','QC-LDPC循环块',r'H_{ij}=P^{a_{ij}}\ \mathrm{or}\ 0,\quad H(x)c(x)^T=0\pmod{x^Z-1}',
    '用Z阶循环置换块表达稀疏校验矩阵。','非二元系数另行定义；等价码表示不必等于发射机内部矩阵。','公开Starlink code5有N=2048、K=1024、Z=128的等价QC关系。',['CCSDS','LIU'])
add('F36','纠错与校验','BCH/RS有限域综合',r'S_j=r(\alpha^{b+j})=\sum_i r_i(\alpha^{b+j})^i',
    '以生成多项式的根检查码字；RS在GF(2^m)符号上运算。','需确定有限域、本原多项式、根序、符号排列和缩短方式。','学习与候选字典，不能因卫星通信常用RS就认定本信号采用RS。',['CCSDS','DVBS2'])
add('F37','纠错与校验','Polar编码',r'G_N=B_NF^{\otimes n},\quad F=\left[\begin{smallmatrix}1&0\\1&1\end{smallmatrix}\right],\quad c=uG_N',
    '通过递归极化变换并冻结部分输入构造码。','具体比特逆序与矩阵约定可不同；需码长、冻结集和速率匹配。','作为全球编码方法学习，不是本地Starlink采用Polar的证据。',['POLAR'])
add('F38','纠错与校验','交织、穿孔与擦除',r'v_i=c_{\pi(i)},\quad c_{obs}=P_{obs}c,\quad L_{erasure}=0',
    '交织改变排列，穿孔/采集缺失只保留部分约束。','擦除不同于硬判0；接收不到的比特不能补成已知值。','现有NPZ仅覆盖少数头部位置，延长同一窄频带不增加缺失列。',['CCSDS','CONV'])
add('F39','统计与边界','信息熵',r'H(B)=-\sum_b p(b)\log_2p(b)',
    '量化一个已定义离散变量的平均不确定性。','采样相关性、有限样本和映射未知会影响估计；高熵不证明加密。','不把看似随机的I/Q或字节当成密文的充分证据。',['SHANNON'])
add('F40','统计与边界','AWGN容量界',r'C=B\log_2(1+S/N)',
    '带宽、信噪比与理论可达信息率的理想关系。','需相应AWGN信道与功率约束；不能把数字录波总功率当作已知SNR。','当前约10%的事件相关功率变化不等于准确的通信SNR。',['SHANNON','MIT'])
add('F41','统计与边界','Fisher信息与估计下界',r'J_{ij}=\frac2{\sigma^2}\Re\{(\partial_{\theta_i}\mu)^H(\partial_{\theta_j}\mu)\},\quad\operatorname{Cov}(\hat\theta)\succeq J^{-1}',
    '在明确观测和噪声模型下约束可达精度。','此式针对独立适当复高斯噪声；有偏估计、模型失配与干扰需另处理。','采样间隔、搜索步长和CZT网格都不是实际测量精度。',['MIT'])
add('F42','统计与边界','整块重采样与条件区间',r'\theta^{(r)}=\operatorname{fit}(\hat y+e_{\mathcal B_r}),\quad I_{.95}=[Q_{.025},Q_{.975}]',
    '保持相关观测成组地重采样，用于估计噪声条件下的参数变化。','区间取决于块长、模型和可交换性；多次重采样不是多次真实事件。','终止沿的细线区间约十几到几十毫秒，未达到微秒级。',['DG94'])
add('F43','Starlink条件参考','PN15参考递推与相位',r'p[n]=p[n-14]\oplus p[n-15],\quad R_{s,k}=e^{j\pi\rho_{s,k}/2}',
    '公开研究用PN15及明确的载波/槽索引生成参考旋转。','LIU v2给出种子000000100000010与两路偏移16327/16320；仍须完整对齐约定。','本地已核对八根导频相位关系；不能把参考符号说成未知消息。',['LIU'])
add('F44','Starlink条件参考','短头部编码块与CRC',r'114=3(32+6),\quad (g_0,g_1,g_2)=(155,123,137)_8',
    '每块32保留输入位及6零尾位，三路输出形成114编码资源。','仅在公开验证的配置下成立；极性、串行顺序、状态与CRC规则均需核对。','第一头部228资源，但本地NPZ每帧仅见两三个位置，仍无CRC通过的头部。',['LIU'])
add('F45','Starlink条件参考','T-code低熵结构候选',r'b_{m,s,k}=t_m[(k-16s)\bmod60]',
    '部分区域的二值变化可由60位模板的平移描述。','k需沿用对应论文的1024点索引与起始对齐；不是所有载荷均满足。','此前15符号重复和稀疏极性不足以宣称恢复完整T-code。',['QIN'])
add('F46','Starlink条件参考','资源线性坐标与分配长度',r'r=1004(s-2)+q,\quad N_{allocation}=N_{words}\left\lceil N_c/\log_2M\right\rceil',
    '把跳过导频和中心空载波的资源按频率及槽序排列。','第二式要求确定调制、码字长度和打包规则；高阶非整字母表须另处理。','公开code5的2048位BPSK码字关系已验证，其他MCS解释保留条件性。',['LIU'])
add('F47','统计与边界','部分观测的可辨识秩',r'y=A_{obs}u\pmod2,\quad\dim\ker A_{obs}=K-\operatorname{rank}(A_{obs})',
    '区分算法尚未找到解与观测约束本身不足。','只说明给定线性模型下的自由度；额外已知约束可改变秩。','本地头部五位置的理想并集秩仅4，涉及输入前缀0..5，不覆盖MCS/长度位。',['CONV','LDPC'])
add('F48','统计与边界','多次尝试下的偶然CRC通过',r'P(\hbox{one or more false passes})\le K\,2^{-m}',
    '在每次错误候选的m位CRC近似均匀时，联合界约束多次搜索风险。','均匀假设未必成立；CRC通过仍需重编码、独立帧和语义一致性验证。','禁止枚举大量映射后只展示一次CRC成功；当前未声称任何消息通过。',['CRC','LIU'])

add('F49','纠错与校验','BCJR/MAP与Turbo外信息',r"L(u_k)=\log\frac{\sum_{u_k=0}\alpha_{k-1}(s')\gamma_k(s',s)\beta_k(s)}{\sum_{u_k=1}\alpha_{k-1}(s')\gamma_k(s',s)\beta_k(s)},\quad L_E=L_{post}-L_{ch}-L_A",
    '按状态对边缘概率求和得到逐比特MAP软信息；Turbo分量译码器交换外信息。','求和遍历与输入标签相符的状态转移；先验、通道、映射和交织必须一致，避免重复计算信息。','学习软译码分支；未确认本地码型，不能直接拿Turbo迭代制造收敛结果。',['MIT','CCSDS'])
add('F50','纠错与校验','BCH/RS译码关键方程',r'\Lambda(z)S(z)\equiv\Omega(z)\pmod{z^{2t}}',
    '由综合求错误定位与错误值多项式，可用Berlekamp–Massey或扩展Euclid，再作根搜索与错误值恢复。','t为设计纠错符号数；多项式倒序、根起点和错误值公式依有限域及码约定变化。','只有获得完整或明确擦除模式的有限域符号后才进入此分支，不从宽带功率直接译码。',['CCSDS','DVBS2'])

metadata={'title':'通信信号分析、编码与解码公式学习库','edition':'1','created_on':'2026-11-20',
          'scope':'可扩展的国际公开一手资料框架；不是全球全部公式或全部文献的穷尽清单。',
          'notation':'j为虚数单位；Fs采样率；alpha循环频率；tau相关时延；Ts采样周期；N/CP等参数须由具体模型定义。',
          'source_policy':'来源包括原论文、作者/大学仓库、官方标准和官方实现。访问限制逐项记录；公式统一改写为本手册记号，不声称每项逐式复刻或原作者首创。',
          'source_count':len(S),'formula_card_count':len(cards),'sources':S,'cards':cards}
(artifact_path('formula_library.json')).write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf8')
intro='''# 通信公式与证据学习库：第1版

目标是让公式参与可复核实验，而不是让公式数量替代证据。本库含50组公式、20项国际公开一手资料入口，不声称穷尽全球通信理论。

学习顺序：F01–F08建立观测模型；F09–F15理解本轮循环结构；F16–F28学习同步、调制和软信息；F29–F38学习纠错；F39–F48约束结论并对照Starlink研究；F49–F50补充MAP/Turbo与BCH/RS译码。

每张卡都给出条件、当前数据如何使用及来源。普通SCF、共轭循环统计和高阶累积量分开；相同循环频率不自动意味着相同符号率、同一硬件或同一卫星。

当前实验的关键推导是F13–F14：原始循环频率随相关时延的偏移，与已有细线频漂模型一致；去频漂后两侧宽谱候选回到约750Hz的整数倍。它支持共享的周期时间结构及频率演化，不是载荷比特解码。
'''
cells=[{'cell_type':'markdown','metadata':{},'source':intro.splitlines(keepends=True)}]
text=[metadata['title']+'（第1版）',metadata['scope'],'公式采用LaTeX记号；Jupyter笔记可直接渲染。','']
for c in cards:
    refs='；'.join(f"[{S[x]['title']}]({S[x]['url']})" for x in c['sources'])
    content=f"## {c['id']} · {c['title']}\n\n$$\n{c['latex']}\n$$\n\n含义：{c['meaning']}\n\n成立条件：{c['conditions']}\n\n当前数据：{c['local_application']}\n\n来源：{refs}\n"
    cells.append({'cell_type':'markdown','metadata':{},'source':content.splitlines(keepends=True)})
    text.extend([f"{c['id']} [{c['group']}] {c['title']}",c['latex'],'含义：'+c['meaning'],'成立条件：'+c['conditions'],'当前数据：'+c['local_application'],'来源：'+'; '.join(S[x]['url'] for x in c['sources']),''])
code="""from pathlib import Path
import json
import sys
root = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p/'project.json').is_file())
sys.path.insert(0, str(root))
from project_paths import legacy_artifact
drift = json.loads(legacy_artifact('cyclic_drift_consistency.json').read_text(encoding='utf8'))
for item in drift['rows']:
    print(item['band'], 'tau=', item['lag_s'],
          'observed offset=', item['observed_raw_offset_hz'],
          'predicted=', item['predicted_from_line_rate_offset_hz'],
          'after correction=', item['dechirped_offset_hz'])
# 注意：0.05Hz为搜索网格间隔，并非绝对校准精度。
"""
cells.append({'cell_type':'code','metadata':{},'source':code.splitlines(keepends=True),'execution_count':None,'outputs':[]})
demo_intro='## 可执行学习实验：卷积码 → 注入错误 → Viterbi → CRC\n\n输入是公开F250示例中已经给出的两个信息字。我们自己编码、制造4处错误，再译码并校验。这是合成教学实验，**不是从本地录波解出了头部，也不是独立复现公开论文的空口接收机**。寄存器方向和输出顺序以脚本为准。\n'
cells.append({'cell_type':'markdown','metadata':{},'source':demo_intro.splitlines(keepends=True)})
demo_code="from pathlib import Path\nimport sys\nroot = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p/'project.json').is_file())\nsys.path.insert(0, str(root/'learning'))\nfrom educational_fec_demo import run_demo\nrun_demo()\n"
cells.append({'cell_type':'code','metadata':{},'source':demo_code.splitlines(keepends=True),'execution_count':None,'outputs':[]})
refs='## 来源核查记录\n\n'+'\n\n'.join(f"**{s['id']}** [{s['title']}]({s['url']}) — {s['verification']}。核查日期：{s['checked_on']}。" for s in S.values())
cells.append({'cell_type':'markdown','metadata':{},'source':refs.splitlines(keepends=True)})
nb={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.11'}},'nbformat':4,'nbformat_minor':5}
# Cell ids make the notebook valid against modern notebook schemas.
for i,c in enumerate(cells):c['id']=f'formula-learning-{i:03}'
(artifact_path('通信公式与证据学习.ipynb')).write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf8')
(artifact_path('通信公式学习索引.txt')).write_text('\n'.join(text),encoding='utf8')
assert len(cards)==50 and len(S)==20
print(f'Built {len(cards)} formula cards with {len(S)} primary-source entries; notebook, JSON and text index saved.')

if __name__=='__main__':pass
