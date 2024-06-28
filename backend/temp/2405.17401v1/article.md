RB-Modulation: Training-Free Personalization of Diffusion Models using Stochastic Optimal Control
=================================================================================================



###### Abstract

We propose Reference-Based Modulation (RB-Modulation), a new plug-and-play solution for training-free personalization of diffusion models.
Existing training-free approaches exhibit difficulties in (a) style extraction from reference images in the absence of additional style or content text descriptions, (b) unwanted content leakage from reference style images, and (c) effective composition of style and content.
RB-Modulation is built on a novel stochastic optimal controller where a style descriptor encodes the desired attributes through a terminal cost.
The resulting drift not only overcomes the difficulties above, but also ensures high fidelity to the reference style and adheres to the given text prompt.
We also introduce a cross-attention-based feature aggregation scheme that allows RB-Modulation to decouple content and style from the reference image.
With theoretical justification and empirical evidence, our framework demonstrates precise extraction and control of content and style in a training-free manner.
Further, our method allows a seamless composition of content and style, which marks a departure from the dependency on external adapters or ControlNets††∗ This work was done during an internship at Google.
.


![](x1.png)

Figure 1:
Given a single reference image (rounded rectangle),
our method RB-Modulation offers a plug-and-play solution for (a) stylization, and (b) content-style composition with various prompts while maintaining sample diversity and prompt alignment.
For instance, given a reference style image (*e.g*. “melting golden 3d rendering style”) and content image (*e.g*. (A) “dog”), our method adheres to the desired prompts without leaking contents from the reference style image and without being restricted to the pose of the reference content image.



1 Introduction
--------------


Text-to-image (T2I) generative models [dalle, dalle2, ldm, imagen] have excelled in crafting visually appealing images from text prompts.
These T2I models are increasingly employed in creative endeavors such as visual arts [imagereward], gaming [pearce2023imitating], personalized image synthesis [dreambooth, consistentid, lora, ziplora], stylized rendering [styledrop, stylealigned, instantstyle, ssa], and image inversion or editing [indi, psld, stsl, nti].
Content creators often need precise control over both the *content* and the *style* of generated images to match their vision.
While the content of an image can be conveyed through text, articulating an artist’s unique style – characterized by distinct brushstrokes, color palette, material, and texture – is substantially more nuanced and complex.
This has led to research on personalization through visual prompting [styledrop, stylealigned, instantstyle].



Recent studies have focused on finetuning pre-trained T2I models to learn style from a set of reference images [gal2022image, dreambooth, styledrop, lora]. This involves optimizing the model’s text embeddings, model weights, or both, using the denoising diffusion loss.
However, these methods demand substantial computational resources for training or finetuning large-scale foundation models, thus making them expensive to adapt to new, unseen styles.
Furthermore, these methods often depend on human-curated images of the same style, which is less practical and can compromise quality when only a single reference image is available.



In training-free stylization, recent methods [stylealigned, instantstyle, ssa] manipulate keys and values within the attention layers using just one reference style image.
These methods face challenges in both extracting the style from the reference style image and accurately transferring the style to a target content image.
For instance, during the DDIM inversion step [ddim] utilized by StyleAligned [stylealigned], fine-grained details tend to be compromised.
To mitigate this issue, InstantStyle [instantstyle] incorporates features from the reference style image into specific layers of a previously trained IP-Adapter [ipadapter].
However, identifying the exact layer for feature injection in a model is complex and not universally applicable across models.
Also, feature injection can cause content leakage from the style image into the generated content.
Moving on to content-style composition, InstantStyle [instantstyle] employs a ControlNet [controlnet] (an additionally trained network) to preserve image layout, which inadvertently limits its diversity.



We introduce Reference-Based Modulation (RB-Modulation), a novel approach for content and style personalization that eliminates the need for training or finetuning diffusion models (*e.g*. ControlNet \citepcontrolnet or adapters \citepipadapter,lora).
Our work reveals that the reverse dynamics in diffusion models can be formulated as stochastic optimal control with a terminal cost.
By incorporating style features into the controller’s terminal cost, we modulate the drift field in diffusion models’ reverse dynamics, enabling training-free personalization.
Unlike conventional attention processors that often leak content from the reference style image, we propose to enhance the image fidelity via an Attention Feature Aggregation (AFA) module that decouples content from reference style image.
We demonstrate the effectiveness of our method in stylization [stylealigned, instantstyle, ssa] and style+content composition, as illustrated in Fig. 1(a) and (b), respectively.
Our experiments show that RB-Modulation outperforms current SoTA methods [stylealigned, instantstyle] in terms of human preference and prompt-alignment metrics.



Our contributions are summarized as follows:

* •

We present reference-based modulation (RB-Modulation), a novel stochastic optimal control framework that enables training-free, personalized style and content control, with a new Attention Feature Aggregation (AFA) module to maintain high fidelity to the reference image while adhering to the given prompt (§4).
* •

We provide theoretical justifications connecting optimal control and reverse diffusion dynamics. We leverage this connection to incorporate desired attributes (*e.g*., style) in our controller’s terminal cost and personalize T2I models in a training-free manner (§5).
* •

We perform extensive experiments covering stylization and content-style composition, demonstrating superior performance over SoTA methods in human preference metrics (§6).



2 Related Work
--------------


Personalization of T2I models:
T2I generative models [ldm, sdxl, sc] are pushing the boundaries in generating aesthetically pleasing images by precisely interpreting text prompts.
Their ability to follow a desired text has unlocked new avenues in personalized content creation, including text-guided image editing [stsl, nti], solving inverse problems [psld, stsl], concept-driven generation [dreambooth, key, multi-concept, chen2024subject], personalized outpainting [tang2023realfill], identity-preservation [ruiz2023hyperdreambooth, consistentid, instantid], and stylized synthesis [styledrop, instantstyle, stylealigned, ziplora].
To tailor T2I models for a specific style (*e.g*., painting) or content (*e.g*., object), existing methods follow one of two recipes: (1) full finetuning (FT) [dreambooth, everaert2023diffusion] or parameter efficient finetuning (PEFT) [multi-concept, ipadapter, lora, styledrop, ziplora] and (2) training(finetuning)-free [stylealigned, instantstyle, ssa], which we discuss below.



Finetuning T2I models for personalization:
FT
 [dreambooth, everaert2023diffusion]
and PEFT
 [multi-concept, ipadapter, lora, styledrop, ziplora]
methods, such as IP-Adapter [ipadapter], LoRA [lora], ZipLoRA [ziplora], and StyleDrop [styledrop], excel at capturing style or object details when the underlying T2I model is finetuned on a few (typically 4) reference images for few thousand iterations.
With the increasing size of T2I models,
PEFT is preferred over FT due to fewer trainable parameters.
However, the challenge of curating a set of reference images and resource-intensive finetuning for every style or content remains largely unexplored.



Training(finetuning)-free T2I models for personalization:
The need to improve T2I model finetuning has sparked interests in training-free personalization methods. In StyleAligned [stylealigned], a reference style image and a text prompt describing the style are used to extract style features via DDIM inversion [ddim].
Target queries and keys are then normalized using adaptive instance normalization [AdaIn] based on reference counterparts.
Finally, reference image keys and values are merged with DDIM-inverted latents in self-attention layers, which tends to leak content information from the reference style image (Figure 3).
Moreover, the need for textual description in the DDIM inversion step can degrade its performance.
Swapping Self-Attention (SSA) [ssa]
addresses these limitations by replacing the target keys and values in self-attention layers with those from a reference style image.
Yet, it still relies on DDIM inversion to cache keys and values of the reference style, which tends to compromise fine-grained details [instantstyle].
Both StyleAligned [stylealigned] and SSA [ssa] require two reverse processes to share their attention layer features and thus demand significant memory.
Recently, InstantStyle [instantstyle] injects reference style features into specific cross-attention layers of IP-Adapter [ipadapter], addressing two key limitations: DDIM inversion and memory-intensive reverse processes.
However, pinpointing the exact layers for feature injection is complex, and they may not generalize to other models.
In addition, when composing style and content, InstantStyle [instantstyle] relies on ControlNet [controlnet], which can limit the diversity of generated images to fixed layouts and deviate from the prompt.



Optimal Control:
Stochastic optimal control finds wide applications in diverse fields such as molecular dynamics [holdijk2024stochastic],
economics [fleming2012deterministic],
non-convex optimization [chaudhari2018deep], robotics [theodorou2011iterative],
and mean-field games [carmona2018probabilistic]
Despite its extensive use, it has been less explored in personalizing diffusion models.
In this paper, we introduce a novel framework leveraging the main concepts from optimal control to achieve training-free personalization.
A key aspect of optimal control is designing a controller to guide a stochastic process towards a desired terminal condition [fleming2012deterministic].
This aligns with our goal of training-free personalization, as we target a specific style or content at the end of the reverse diffusion process, which can be incorporated in the controller’s terminal condition.



RB-Modulation overcomes several challenges encountered by SoTA methods [stylealigned, ssa, instantstyle].
Since RB-Modulation does not require DDIM inversion, it retains fine-grained details unlike StyleAligned [stylealigned].
Using a stochastic controller to refine the trajectory of a single reverse process, it overcomes the limitation of coupled reverse processes [stylealigned].
By incorporating a style descriptor in our controller’s terminal cost, it eliminates the dependency on Adapters [ipadapter, lora] or ControlNets [controlnet] by InstantStyle [instantstyle].



3 Preliminaries
---------------


Diffusion models consist of two stochastic processes:
(a) noising process, modeled by a Stochastic Differential Equation (SDE) known as forward-SDE:
$\mathrm{d}X\_{t}=f(X\_{t},t)\,\mathrm{d}t+g(X\_{t},t)\,\mathrm{d}W\_{t},X\_{0}\sim p%
\_{0}$,
and (b) denoising process, modeled by the time-reversal of forward-SDE under mild regularity conditions [anderson], also known as reverse-SDE:



|  | $\displaystyle\mathrm{d}X\_{t}=\left[f(X\_{t},t)-g^{2}(X\_{t},t)\nabla\log p(X\_{t}% ,t)\right]\,\mathrm{d}t+g(X\_{t},t)\,\mathrm{d}W\_{t},\qquad X\_{1}\sim{\mathcal{% N}}\left(0,\mathrm{I}\_{d}\right).$ |  | (1) |
| --- | --- | --- | --- |

Here, $W=(W\_{t})\_{t\geq 0}$ is standard Brownian motion in a filtered probability space, $(\Omega,\mathcal{F},(\mathcal{F}\_{t})\_{t\geq 0},\mathcal{P})$,
$p(\cdot,t)$ denotes the marginal density of $p$ at time $t$, and $\nabla\log p\_{t}(\cdot)$ the corresponding score function.
$f(X\_{t},t)$ and $g(X\_{t},t)$
are called drift and volatility, respectively.
A popular choice of $f(X\_{t},t)=-X\_{t}$ and $g(X\_{t},t)=\sqrt{2}$ corresponds to the well-known forward Ornstein-Uhlenbeck (OU) process.



For T2I generation, the reverse-SDE (1) is simulated using a neural network $s\left({\mathbf{x}}\_{t},t;\theta\right)$
[hyvarinen2005estimation, dsm] to approximate $\nabla\_{\mathbf{x}}\log p({\mathbf{x}}\_{t},t)$.
Importantly, to accelerate the sampling process in practice [ddim, edm, zhang2022fast], the reverse-SDE (1) shares the same path measure with a probability flow ODE:
$\mathrm{d}X\_{t}=\left[f(X\_{t},t)-\frac{1}{2}g^{2}(X\_{t},t)\nabla\log p(X\_{t},t%
)\right]\,\mathrm{d}t$, where $X\_{1}\sim{\mathcal{N}}\left(0,\mathrm{I}\_{d}\right)$.



Personalized diffusion models either fully finetune $\theta$ of $s\left({\mathbf{x}}\_{t},t;\theta\right)$ [dreambooth, everaert2023diffusion], or train a parameter-efficient adapter $\Delta\theta$ for $s\left({\mathbf{x}}\_{t},t;\theta+\Delta\theta\right)$ on reference style images [lora, styledrop, ziplora].
Our method does not finetune $\theta$ or train $\Delta\theta$.
Instead, we derive a new drift field through a stochastic optimal controller that modulates the drift of the standard reverse-SDE (1).



4 Method
--------


Personalization using optimal control:
Normalize time $t$ by the total number of diffusion steps $T$ such that $0\leq t\leq 1$.
Let us denote by $u:\mathbb{R}^{d}\times[0,1]\rightarrow\mathbb{R}^{d}$ a controller
from the admissible set of controls
${\mathcal{U}}\subseteq\mathbb{R}^{d}$,
$X\_{t}^{u}\in\mathbb{R}^{d}$ a state variable,
$\ell:\mathbb{R}^{d}\times\mathbb{R}^{d}\times[0,1]\to\mathbb{R}$ the transient cost, and $h:\mathbb{R}^{d}\to\mathbb{R}$ the terminal cost of the reverse process $(X\_{t}^{u})\_{t=1}^{0}$.
We show in §5 that training-free personalization can be formulated as a control problem where the drift of the standard reverse-SDE (1) is modified via RB-modulation:



|  |  | $\displaystyle\min\_{u\in\mathcal{U}}\mathbb{E}\Big{[}\int\_{1}^{0}\ell\left(X^{u% }\_{t},u(X^{u}\_{t},t),t\right)\mathrm{d}t+\gamma h(X^{u}\_{0})\Big{]},\quad\text% {where}$ |  | (2) |
| --- | --- | --- | --- | --- |
|  |  | $\displaystyle\mathrm{d}X^{u}\_{t}=\left[f(X\_{t}^{u},t)-g^{2}(X\_{t}^{u},t)\nabla% \log p(X\_{t}^{u},t)+u(X^{u}\_{t},t)\right]\mathrm{d}t+g(X^{u}\_{t},t)\mathrm{d}W% \_{t},X^{u}\_{1}\sim{\mathcal{N}}\left(0,\mathrm{I}\_{d}\right).$ |  |
| --- | --- | --- | --- |

Importantly, the terminal cost $h(\cdot)$, weighted by $\gamma$, captures the discrepancy in feature space between the styles of the reference image and the generated image.
The resulting controller $u(\cdot,t)$ modulates the drift over time to satisfy this terminal cost. We derive the solution to this optimal control problem through the
Hamilton-Jacobi-Bellman (HJB) equation [fleming2012deterministic]; refer to Appendix A for details.
Our proposed RB-Modulation Algorithm 1 has two key components: (a) stochastic optimal controller and (b) attention feature aggregation. Below, we discuss each in turn.



(a) Stochastic Optimal Controller (SOC):
We show that the reverse dynamics in diffusion models can be framed as a stochastic optimal control problem with a quadratic terminal cost (theoretical analysis in §5).
For personalization using a reference style image $X\_{0}^{f}={\mathbf{z}}\_{0}$, we use a Consistent Style Descriptor (CSD) [csd] to extract style features $\Psi(X\_{0}^{f})$.
Since the score functions $s\left({\mathbf{x}}\_{t},t;\theta\right)\!\approx\!\nabla\log p\left(X\_{t},t\right)$ are available from pre-trained diffusion models [sdxl, sc], our goal is to add a correction term $u(\cdot,t)$ to modulate the reverse-SDE and minimize the overall cost (2).
We approximate $X\_{0}^{u}$ with its conditional expectation using Tweedie’s formula [psld, stsl].
Finally, we incorporate the style features into our controller’s terminal cost as: $h\left(X^{u}\_{0}\right)=\lVert\Psi(X^{f}\_{0})-\Psi(\mathbb{E}\left[X^{u}\_{0}|X%
^{u}\_{t}\right])\rVert^{2}\_{2}$.



Our theoretical results (§5) suggest that the optimal controller can be obtained by solving the HJB equation and letting $\gamma\rightarrow\infty$. In practice, this translates to dropping the transient cost $\ell\left(X^{u}\_{t},u(X^{u}\_{t},t),t\right)$ and solving (2) with only the terminal constraint, *i.e*.,



|  |  | $\displaystyle\min\_{u\in\mathcal{U}}\lVert\Psi(X^{f}\_{0})-\Psi(\mathbb{E}\left[% X^{u}\_{0}|X^{u}\_{t}\right])\rVert^{2}\_{2}.$ |  | (3) |
| --- | --- | --- | --- | --- |

Thus, we solve (3) to find the optimal control $u$ and use this controller in the reverse dynamics (2) to update the current state from $X^{u}\_{t}$ to $X^{u}\_{t-\Delta t}$ (recall that time flows backwards in the reverse-SDE (1)). Our implementation of (3) is given in Algorithm 1, which follows from our theoretical insights.





Input: Diffusion steps $T$,
reference prompt ${\mathbf{p}}$,
reference image ${\mathbf{z}}\_{0}$,
style descriptor $\Psi(\cdot)$,
score network $s(\cdot,\cdot,\cdot;\theta)$


Tunable parameter: Stepsize $\eta$, optimization steps $M$



Output: Personalized latent $X^{u}\_{0}$

1
Initialize ${\mathbf{x}}\_{T}\leftarrow{\mathcal{N}}\left(0,\mathrm{I}\_{d}\right)$

2
for *$t=T$ to $1$* do

3      
Initialize controller $u=0$

4      
for *$m=1$ to $M$* do

5            
$\hat{{\mathbf{x}}}\_{t}={\mathbf{x}}\_{t}+u$
$\triangleright$ controlled state

6            
$\bar{X}^{u}\_{0}=\frac{\hat{{\mathbf{x}}}\_{t}}{\sqrt{\bar{\alpha}\_{t}}}+\frac{(%
1-\bar{\alpha}\_{t})}{\sqrt{\bar{\alpha}\_{t}}}s\left(\hat{{\mathbf{x}}}\_{t},t,{%
\mathbf{p}};\theta\right)$


7            
$h(\bar{X}^{u}\_{0})=\lVert\Psi({\mathbf{z}}\_{0})-\Psi(\bar{X}^{u}\_{0})\rVert^{2%
}\_{2}$ using Eq. (3)

8            
$u=u-\eta\nabla\_{u}h(\bar{X}^{u}\_{0})$
$\triangleright$ update controller


9       end for

10      ${\mathbf{x}}^{\*}\_{t}={\mathbf{x}}\_{t}+u$
$\triangleright$ optimally controlled state


11      
$\bar{X}^{u}\_{0}=\frac{{\mathbf{x}}^{\*}\_{t}}{\sqrt{\bar{\alpha}\_{t}}}+\frac{(1-%
\bar{\alpha}\_{t})}{\sqrt{\bar{\alpha}\_{t}}}s\left({\mathbf{x}}^{\*}\_{t},t,{%
\mathbf{p}};\theta\right)$
$\triangleright$ terminal state


12      
${\mathbf{x}}\_{t-1}\leftarrow\text{DDIM}(\bar{X}^{u}\_{0},{\mathbf{x}}^{\*}\_{t})$
$\triangleright$ one step reverse-SDE [ddim]


13 end for

return $X^{u}\_{0}$



Algorithm 1 Reference-Based Modulation



![](x2.png)

Figure 2: Attention Feature Aggregation (AFA):
Within the cross-attention layers, the keys and values from the previous layers ($K$,$V$), text embedding ($K\_{p}$,$V\_{p}$), reference style image ($K\_{s}$,$V\_{s}$) and reference content image ($K\_{c}$,$V\_{c}$) are concatenated and processed separately to disentangle the information, which is followed by an averaging layer for the output. $K\_{c}$,$V\_{c}$ and only used for content-style composition.



Implementation challenge:
For smaller generative models [ldm], we can directly solve our control problem (3).
However, for larger models [sdxl, sc], optimizing our control objective (3) requires back propagation through the score network $s\left({\mathbf{x}}\_{t},t;\theta\right)$ with tentatively billions of parameters.
This significantly increases time and memory complexity [psld, stsl].



We propose a proximal gradient descent approach to address this challenge.
Recall that the key ingredient of our Algorithm 1 is to find the previous state $X\_{t-\Delta t}$ by modulating the current state $X\_{t}$ based on an optimal controller $u^{\*}$.
The optimal controller $u^{\*}$ is obtained by minimizing the discrepancy in style between $\bar{X}^{u}\_{0}\coloneqq\mathbb{E}\Big{[}X^{u}\_{0}|X^{u}\_{t}={\mathbf{x}}\_{t}%
\Big{]}$, obtained using our controlled reverse-SDE (3), and the reference style image ${\mathbf{z}}\_{0}$.
Motivated by this interpretation, an alternate Algorithm 2 (see Appendix B.2) avoids back propagation through $s({\mathbf{x}}\_{t},t;\theta)$ by introducing a dummy variable ${\mathbf{x}}\_{0}$, which serves as a proxy for $\bar{X}\_{0}^{u}$ in the terminal cost.
Instead of forcing ${\mathbf{x}}\_{0}$ to be decided by the dynamics of the reverse-SDE as in Algorithm 1, we allow it to be only approximately faithful to the dynamics.
This is implemented by adding a proximal penalty, *i.e*. ${\mathbf{x}}^{\*}\_{0}=\operatorname\*{arg\,min}\_{{\mathbf{x}}\_{0}\in\mathbb{R}^{%
d}}\lVert\Psi(X^{f}\_{0})-\Psi({\mathbf{x}}\_{0})\rVert^{2}\_{2}+\lambda\lVert{%
\mathbf{x}}\_{0}-\mathbb{E}\left[X^{u}\_{0}|X^{u}\_{t}\right]\rVert\_{2}^{2}$,
where the hyper-parameter $\lambda$ controls the faithfulness of the reverse dynamics.
This penalty assumes that with a small step-size in the reverse-SDE dynamics (3), ${\mathbf{x}}^{\*}\_{0}$ and $\mathbb{E}\Big{[}X^{u}\_{0}|X^{u}\_{t}={\mathbf{x}}\_{t}\Big{]}$ will be close.
Therefore, Algorithm 2 enables personalization of large-scale foundation models without significantly increasing time and memory complexity.



(b) Attention Feature Aggregation (AFA):
Transformer-based diffusion models [ldm, sdxl, sc] consist of self-attention and cross-attention layers operating on latent embedding ${\mathbf{x}}\_{t}\in\mathbb{R}^{d\times n\_{h}}$.
Within the attention module $\text{Attention}(Q,K,V)$, ${\mathbf{x}}\_{t}$ is projected into queries $Q\in\mathbb{R}^{d\times n\_{q}}$, keys $K\in\mathbb{R}^{d\times n\_{q}}$, and values $V\in\mathbb{R}^{d\times n\_{h}}$ using linear projections.
Through $Q$, $K$, and $V$, attention layers capture global context and improve long-range dependencies within ${\mathbf{x}}\_{t}$.



To accommodate a reference image (*e.g*., style or content) while retaining prompt-alignment, we propose an Attention Feature Aggregation (AFA) module, as illustrated in Figure 2.
For a given prompt ${\mathbf{p}}$, a reference style image $I\_{s}$, and a reference content image $I\_{c}$, we first extract the embeddings using CLIP [clip] text and image encoder, respectively.
Then, we project these embeddings into keys and values using linear projection layers.
We denote by $K\_{p}$ and $V\_{p}$ the keys and values from ${\mathbf{p}}$, $K\_{s}$ and $V\_{s}$ from $I\_{s}$, $K\_{c}$ and $V\_{c}$ from $I\_{c}$ (used only in content-style composition).
The query $Q$ is obtained from a linear projection of ${\mathbf{x}}\_{t}$, and remains the same in the AFA module.
By processing the keys and values separately,
we disentangle their relative importance with respect to the state variable.
This ensures that the attention maps from text are not contaminated with attention maps from style.
To make the text consistent with the style, we also compose the keys and values of both text and style in our attention processor.
The final output of our AFA module is given by



|  | $\displaystyle AFA=\text{Avg}\left(A\_{text},A\_{style},A\_{text+style}\right),A\_{% text}=\text{Attention}(Q,[K;K\_{p}],[V;V\_{p}]),$ |  |
| --- | --- | --- |
|  | $\displaystyle A\_{style}=\text{Attention}(Q,[K;K\_{s}],[V;V\_{s}]),A\_{text+style}% =\text{Attention}(Q,[K;K\_{p};K\_{s}],[V;V\_{p};V\_{s}]),$ |  |
| --- | --- | --- |

where $[K;K\_{p}]\in\mathbb{R}^{2d\times n\_{q}}$ indicates concatenation of $K$ with $K\_{p}$ along the number of tokens dimension.
For style-content composition, we process the content image $I\_{c}$ in the same way as the reference style image $I\_{s}$, and obtain another set of attention outputs:



|  | $\displaystyle AFA=\text{Avg}\left(A\_{text},A\_{style},A\_{content},A\_{content+% style}\right),$ |  |
| --- | --- | --- |
|  | $\displaystyle A\_{content}=\text{Attention}(Q,[K;K\_{c}],[V;V\_{c}]),A\_{content+% style}=\text{Attention}(Q,[K;K\_{s};K\_{c}],[V;V\_{s};V\_{c}]).$ |  |
| --- | --- | --- |

Importantly, the AFA module is computationally tractable as it only requires the computation of a multi-head attention, which is widely used in practice [sdxl].



5 Theoretical Justifications
----------------------------


Problem setup:
We outline an approach to derive the optimal controller for a special case of our control problem (2).
We substitute $t\leftarrow 1-t$ to account for the time reversal in the reverse-SDE (1).
Here, $X^{u}\_{0}\sim{\mathcal{N}}\left(0,\mathrm{I}\_{d}\right)$ and $X^{u}\_{1}\sim p\_{data}$.
We consider the dynamic without the Brownian motion:
$\mathrm{d}X\_{t}^{u}=v(X\_{t}^{u},u,t)\mathrm{d}t,~{}~{}~{}X\_{t\_{0}}^{u}={%
\mathbf{x}}\_{0},$
where $0\leq t\_{0}\leq t\leq t\_{N}\leq 1$ and $v:\mathbb{R}^{d}\times\mathbb{R}^{d}\times[t\_{0},t\_{N}]\rightarrow\mathbb{R}^{d}$ denotes the drift field.
The optimal controller $u^{\*}$ can be derived by solving the Hamilton-Jacobi-Bellman (HJB) equation [fleming2012deterministic, basar2020lecture],
see Appendix A for details.



Incorporating optimal control in diffusion:
Following recent works \citepkappen2008stochastic,bridge, we consider a dynamical system whose drift field minimizes a transient trajectory cost and a terminal cost (weighted by $\gamma$) to ensure “closeness” to reference content $x\_{1}$ (Appendix A.1).
Proposition A.2 \citepbridge outlines the optimal control in the limiting setting where $\gamma\rightarrow\infty$.
Furthermore, suppose we replace $x\_{1}$ with its conditional expectation (discussed in Remark A.3), the resulting dynamic is the standard reverse-SDE for the Orstein-Uhlenbeck (OU) diffusion process for a particular noise schedule.
This connection between classic linear quadratic control and the standard reverse-SDE allows us to study other diffusion problems (*e.g*., personalization) through the lens of stochastic optimal control.
For instance, we derive the optimal controller given reference style features $y\_{1}$ at the terminal time.



###### Proposition 5.1.


Suppose $A\in\mathbb{R}^{k\times d}$ be a linear style extractor that operates on the terminal state $X^{u}\_{1}\in\mathbb{R}^{d}$.
Given reference style features $y\_{1}$, consider the control problem:



|  | $\displaystyle\min\_{u\in{\mathcal{U}}}\int\_{t\_{0}}^{1}\frac{1}{2}\left\|u(X^{u}% \_{t},t)\right\|^{2}dt+\frac{\gamma}{2}\left\|AX^{u}\_{1}-y\_{1}\right\|\_{2}^{2},% ~{}\text{where }\mathrm{d}X^{u}\_{t}=u(X^{u}\_{t},t)\,\mathrm{d}t,~{}X^{u}\_{t\_{0% }}=x\_{0}.$ |  |
| --- | --- | --- |

Then, in the limit when $\gamma\rightarrow\infty$, the optimal controller $u^{\*}=\frac{\left(A^{T}A\right)^{-1}A^{T}\left(y\_{1}-A{\mathbf{x}}\_{t}\right)}%
{1-t}$, which yields the following controlled dynamic:
$\mathrm{d}X^{u}\_{t}=\frac{\left(A^{T}A\right)^{-1}A^{T}\left(y\_{1}-A{\mathbf{x%
}}\_{t}\right)}{1-t}\mathrm{d}t.$




Implication.
The optimal controller depends on the reference style features $y\_{1}$ at the terminal time, instead of the image content encoded in $x\_{1}$.
To simulate the controlled dynamic in practice,
we use CSD \citepcsd as a style feature extractor and replace $y\_{1}$ with the style features extracted from the expected terminal state $\mathbb{E}\Big{[}X^{u}\_{1}|X^{u}\_{t}\Big{]}$, as discussed in Appendix A.2.



Drift modulation through optimal controller:
We then study a control problem where the velocity field is a linear combination of the state and the control variable.
This problem is interesting to study because
the reverse-SDE dynamic of the standard OU process has a drift field of the form:
$v\left(X\_{t},t\right)=-X\_{t}-2\nabla\log p(X\_{t},t).$
For a Gaussian prior $X\_{0}\sim{\mathcal{N}}\left(0,\mathrm{I}\right)$, the law of the OU process satisfies $\nabla\log p\left(X\_{t},t\right)=-X\_{t}$, and the corresponding drift field becomes $v\left(X\_{t},t\right)=X\_{t}$. Our goal is to modulate this drift field using a controller $u\left(X^{u}\_{t},t\right)$. The result below provides the structure of the optimal control (again in the setting where the terminal objective is known; see Appendix A1).



###### Proposition 5.2.


Suppose $A\in\mathbb{R}^{k\times d}$ be a linear style extractor that operates on the terminal state $X^{u}\_{1}\in\mathbb{R}^{d}$.
Let ${\mathbf{p}}\_{t}$ denote $\nabla\_{\mathbf{x}}V^{\*}({\mathbf{x}},t)$ in HJB equation (A.1).
Given reference style features $y\_{1}$, consider the control problem:



|  | $\displaystyle\min\_{u\in{\mathcal{U}}}\int\_{t\_{0}}^{1}\frac{1}{2}\left\|u(X^{u}% \_{t},t)\right\|^{2}dt+\frac{\gamma}{2}\left\|AX^{u}\_{1}-y\_{1}\right\|\_{2}^{2},% ~{}\text{where }\mathrm{d}X^{u}\_{t}=\Big{[}X^{u}\_{t}+u(X^{u}\_{t},t)\Big{]}\,% \mathrm{d}t,~{}X^{u}\_{t\_{0}}=x\_{0},$ |  |
| --- | --- | --- |

Then, the optimal controller becomes $u^{\*}(t)=-{\mathbf{p}}\_{t}$, where the instantaneous state $X^{u}\_{t}={\mathbf{x}}\_{t}$ and ${\mathbf{p}}\_{t}$ satisfy the following coupled transitions:



|  | $\displaystyle\begin{bmatrix}{\mathbf{x}}\_{t}\\ {\mathbf{p}}\_{t}\end{bmatrix}=\begin{bmatrix}x\_{0}e^{t}-\frac{\gamma}{2}A^{T}% \left(A{\mathbf{x}}\_{1}-y\_{1}\right)e^{1+t}+\frac{\gamma}{2}A^{T}\left(A{% \mathbf{x}}\_{1}-y\_{1}\right)e^{1-t}\\ \gamma A^{T}\left(A{\mathbf{x}}\_{1}-y\_{1}\right)e^{1-t}\end{bmatrix}.$ |  |
| --- | --- | --- |



Summary.
We build on the connection between optimal control and reverse diffusion (see Appendices A.1-A.3 for details). The general strategy is to derive the optimal controller with known terminal state, and then replace the terminal state in the controller with its estimate using Tweedie’s formula. For stylized models and Gaussian prior, the controllers have an explicit form. However in practice, the data distribution may not be Gaussian, and thus, we do not aim for a closed-form expression to modulate the drift.
This line of analysis, however, points to our method RB-Modulation. As discussed in §4, we incorporate a style descriptor in our controller’s terminal cost and numerically evaluate the resulting drift at each reverse time step either through back propagating through the score network (Algorithm 1), or an approximation based on proximal gradient updates (Algorithm 2).


![](x3.png)

Figure 3: Qualitative results for stylization:
A comparison with state-of-the-art methods (InstantStyle [instantstyle], StyleAligned [stylealigned], StyleDrop [styledrop]) highlights our advantages in preventing information leakage from the reference style and adhering more closely to desired prompts.




6 Experiments
-------------


Metrics:
Evaluating stylized synthesis is challenging due to the subjective nature of style, making simple metrics inadequate.
We follow a two step approach: first using metrics from prior works and then conducting human evaluation.
To evaluate prompt-image alignment, we use CLIP-T score [stylealigned, styledrop, instantstyle] and ImageReward [imagereward], which also consider human aesthetics, distortions, and object completeness.
When a style description is provided, CLIP-T and ImageReward also capture style alignment.
We assess style similarity using DINO [dino] and content similarity using CLIP-I [clip] as in prior work [stylealigned, dreambooth, styledrop], and highlight their limitations in disentangling style and content performance in evaluation.
Given the importance of human evaluation in T2I personalization [stylealigned, styledrop, dreambooth, ziplora, ssa], we also conduct a user study though Amazon Mechanical Turk to measure both style and text alignment.



Datasets and baselines:
We use style images from StyleAligned benchmark \citepstylealigned for stylization and content images from DreamBooth [dreambooth] for content-style composition.
We base RB-Modulation on the recently released StableCascade [sc].
We compare our approach with three training-free methods: InstantStyle \citepinstantstyle (state-of-the-art), IP-Adapter \citepipadapter, and StyleAligned \citepstylealigned. For completeness, we also
compare with training-based methods StyleDrop \citepstyledrop and ZipLoRA \citepziplora.



Implementation details:
All experiments run on a single A100 NVIDIA GPU.
We use the same hyper-parameters for our method across tasks, and default settings for alternative methods as per their original papers.
Details are provided in Appendix B.1.



### 6.1 Image Stylization


Qualitative analysis:
This section describes image stylization experiments using a text prompt and a reference style image.
Figure 3 compares our method with SoTA training-free InstantStyle [instantstyle] and StyleAligned [stylealigned], and training-based StyleDrop [styledrop].
Except for StyleDrop, which requires $\sim$5 minutes of training per style, all methods, including ours, are training-free and complete inference in $<$1 minute.
While all methods produce reasonable outputs, alternative methods encounter issues with information leakage.
For instance, in the third row of Figure 3, StyleAligned and StyleDrop generate a wine bottle and book resembling the smartphone in the reference style image.
In the last row, StyleAligned leaks the house and the background of the reference image; InstantStyle exhibits color leakage from the house, resulting in similar-colored images.
Our method accurately adheres to the prompt in the desired style.
As illustrated in the second and the third row, our method generates only one glass of wine and a high-fidelity rubber duck, compared to baselines where extra items appear (wine bottles styled like the left smartphone) or incorrect styles (cartoon-style rubber duck).



{tabu}

to @l@ \*9X[c]@
Human Ours *vs*. InstantStyle [instantstyle] Ours *vs*. StyleAligned [stylealigned] Ours *vs*. IP-Adapter [ipadapter]

Preference (%) OQ $\uparrow$ SA $\uparrow$ PA $\uparrow$ OQ $\uparrow$ SA $\uparrow$ PA $\uparrow$ OQ $\uparrow$ SA $\uparrow$ PA $\uparrow$

Alternative 39.8 38.5 39.5 24.4 27.8 29.4 8.1 20.1 8.3

Tie 9.3 6.4 7.3 8.8 7.1 5.8 6.9 4.8 4.5

RB-Modulation (ours) 51.0 55.1 53.3 66.9 65.1 64.9 85.0 75.1 87.2






Table 1: User study:
We report the % of human preference on ours *vs*. alternatives for overall quality (OQ), style alignment (SA), and prompt alignment (PA), including ties where users couldn’t decide.
Our method consistently outperforms alternatives, achieving higher scores in all metrics.



User study:
To validate the qualitative analysis, we conduct a user study on Amazon Mechanical Turk with 155 participants using 100 styles from the StyleAligned dataset [stylealigned], collecting a total of 7,200 answers (8 responses for each question).
Each user answers 3 questions comparing our method with an alternative method regarding (1) overall quality, (2) style alignment, and (3) prompt alignment (details in the Appendix B.7).
Table 6.1 summarizes the percentage of human preferences for our method, the alternative method, or a tie.
Our method consistently outperforms the alternatives, including the most competitive method InstantStyle [instantstyle] in style alignment.
The preference rates over all three metrics highlight the effectiveness of our method RB-Modulation.



Quantitative analysis:
Table 6.1 evaluates 300 prompts and 100 styles on the StyleAligned dataset [stylealigned] using three metrics, with and without style descriptions in the prompts.
Our method outperforms others notably in the ImageReward metric, closely matching human aesthetics assessment from the user study in Table 6.1.
In addition, the CLIP-T score indicates
our effective alignment between generated images and text prompts.
While IP-Adapter and StyleAligned have higher DINO scores, their lower rating in ImageReward, CLIP-T and user preference expose information leakage from the reference style images.
Nevertheless, our DINO score remains competitive with the leading method InstantStyle.
Notably, all metrics show improvement with style descriptions, particularly in ImageReward, where leveraging style descriptions enhances prompt alignment.
Our method achieves high ImageReward and CLIP-T score even without style descriptions, suggesting robustness in prompt alignment without explicit style information in the prompt.



![](x4.png)

Figure 4: Ablation study:
Our method builds on any transformer-based diffusion model.
In this case, we use StableCascade [sc] as the foundation, and sequentially add each module to show their effectiveness.
DirectConcat involves concatenating reference image embeddings with prompt embeddings.
Style descriptions are excluded in this ablation study.





Ablation Study:
Figure 4 shows an ablation study of the AFA and SOC modules
adding new capabilities to StableCascade [sc].
We include a baseline, “DirectConcat”, which concatenates reference style embeddings with text embeddings in the cross-attention modules.
DirectConcat mixes both embeddings, making it less effective in disentangling style from prompts (*e.g*., cat *vs*. lighthouse).
While AFA or SOC alone mitigates this by modulating the reverse drift and attention modules (§4), each has drawbacks.
AFA alone fails to capture the cat’s style accurately, and SOC alone misplaces elements, like “a lighthouse hat on the cat” and “a railroad trunk on a piano”.
We observe consistent improvements with each module, with the best results when combined.
Quantitative analysis is omitted due to the lack of suitable metrics for information leakage, as detailed in Appendix B.5.





{tabu}

to @l@    \*6X[c]@
 ImageReward $\uparrow$ CLIP-T score $\uparrow$ DINO score

With style description? No Yes No Yes No Yes

IP-Adapter [ipadapter] -1.99 -1.51 0.21 0.26 0.89 0.89

StyleAligned [stylealigned] -0.68 0.01 0.26 0.31 0.80 0.85

InstantStyle [instantstyle] 0.09 0.72 0.29 0.33 0.68 0.72

RB-Modulation (ours) 0.91 1.18 0.30 0.34 0.68 0.73






Table 2: Quantitative results for stylization:
We compare alternative methods on three metrics: ImageReward [imagereward] and CLIP-T [clip] for prompt alignment, DINO [dino] for style alignment.
Note that DINO score does not capture information leakage,
so higher scores are not necessarily better (§B.5).


![](x5.png)

Figure 5: Qualitative results for content-style composition:
Our method shows better prompt alignment and greater diversity than training-free methods IP-Adapter [ipadapter] and InstantStyle [instantstyle], and have competitive performance with training-based ZipLoRA [ziplora] .





### 6.2 Content-Style Composition


Qualitative analysis:
Content-style composition aims to preserve the essence of both content and style depicted in the reference images, while ensuring the resulting image aligns with a given text prompt.
Figure 5 compares our method against training-free InstantStyle [instantstyle], IP-Adapter [ipadapter], and training-based ZipLoRA [ziplora].
Notably, the training-free InstantStyle and IP-Adapter rely on ControlNet [controlnet], which often constrains their ability to accurately follow prompts for changing the pose of the generated content, such as illustrating “dancing” in Figure 5(b), or “walking” in (c).
In contrast, our method avoids the need for ControlNet or adapters, and can effectively capture the distinctive attributes of both style and content images while adhering to the prompt to generate diverse images.
In Figure 5(a), our method accurately captures elements like “table” and “river” that are overlooked in InstantStyle and IP-Adapter.
In addition, our method mitigates information leakage, as evidenced in Figure 5(b), where the trunk of the tree behind the sloth is erroneously captured by InstantStyle and IP-Adapter but not by ours.
Compared to ZipLoRA [ziplora] that requires training of 12 LoRAs \citeplora and additional merge layers for each composition, our method requires no training at all while yielding competitive or better results.
For instance, our method effectively captures the 2D cartoon and 3D rendering styles as illustrated in Figures 5(a) and (b).



{tabu}

to @l@    \*4X[c]@
 ImageReward $\uparrow$ CLIP-T score $\uparrow$ DINO score
CLIP-I score

IP-Adapter [ipadapter] -0.78 0.22 0.73 0.68

InstantStyle [instantstyle] -0.54 0.21 0.71 0.71

RB-Modulation (ours) 0.74 0.26 0.74 0.71






Table 3: Quantitative results for composition:
In addition to the metrics used for stylization, we use CLIP-T score [clip] to evaluate content alignment with the reference content. Similar to DINO, CLIP-I could inflate test score [styledrop, ziplora] by capturing content leakage, but not necessarily preferred by users; higher DINO and CLIP-I scores do not mean better human preference.



Quantitative analysis:
Table 6.2 shows quantitative evaluation using 50 styles from StyleAligned dataset [stylealigned] and 5 contents from DreamBooth dataset [dreambooth].
Unlike prior works [stylealigned, styledrop, ziplora, dreambooth, ssa] reporting either DINO and CLIP-I scores, we present both metrics and demonstrate comparable performance across them.
Additionally, we obtain notably higher ImageReward score, which aligns closely with human aesthetics assessment as evidenced in §6.1 and [imagereward].
Consequently, we omitted a user study in this section.
For more details, please refer to Appendix B.1.




7 Conclusion
------------


We introduced Reference-Based modulation (RB-Modulation), a training-free method for personalizing transformer-based diffusion models.
RB-Modulation builds on concepts from stochastic optimal control to modulate the drift field of reverse diffusion dynamics, incorporating desired attributes (*e.g*., style or content) via a terminal cost.
Our Attention Feature Aggregation (AFA) module decouples content and style in the cross-attention layers and enables precise control over both.
In addition, we derived theoretical connections between linear quadratic control and the denoising diffusion process, which led to the creation of RB-Modulation.
Empirically, our method outperformed current state-of-the-art methods in stylization and content+style composition.
To our best knowledge, this is the first training-free personalization framework using stochastic optimal control,
which marks the departure from external adapters or ControlNets.



Limitation:
We proposed a framework and demonstrated its efficacy by incorporating a style descriptor [csd] in a pre-trained diffusion model [sc].
The inherent limitations of the style descriptor or diffusion model might propagate into our framework.
We believe these limitations can be addressed by appropriate replacements of the descriptor or generative prior in a plug-and-play manner.



Acknowledgements:
This research has been supported by NSF Grant 2019844, a Google research collaboration award, and the UT Austin Machine Learning Lab.
Litu Rout has been supported by Ju-Nam and Pearl Chew Presidential Fellowship and George J. Heuer Graduate Fellowship.



\printbibliography


Appendix A Additional Theoretical Results
-----------------------------------------


In this section, we restate the propositions more precisely and provide their technical proofs. First, we recall standard terminologies from optimal control literature [fleming2012deterministic]. For $0\leq t\_{0}\leq t\leq t\_{N}\leq 1$, the cost function associated with the controller $u(\cdot)$ is defined by the integral:



|  | $\displaystyle V(u;{\mathbf{x}}\_{0},t\_{0})=\int\_{t\_{0}}^{t\_{N}}\ell\left(X\_{t}^% {u},u,t\right)dt+h\left(X\_{t\_{N}}^{u}\right),~{}~{}~{}X\_{t\_{0}}^{u}={\mathbf{x% }}\_{0},$ |  | (4) |
| --- | --- | --- | --- |

where $\ell(\cdots)$ denotes a scalar valued function of the state $X\_{t}^{u}$, controller $u(\cdot)$, and instantaneous time $t$. The value function $V^{\*}({\mathbf{x}}\_{0},t\_{0})$ is defined as the minimum value of $V(u;{\mathbf{x}}\_{0},t\_{0})$ over the set of admissible controllers ${\mathcal{U}}$, i.e.,



|  | $\displaystyle V^{\*}=V^{\*}({\mathbf{x}}\_{0},t\_{0})=\min\_{u\in{\mathcal{U}}}V(u;% {\mathbf{x}}\_{0},t\_{0})=\min\_{u\in{\mathcal{U}}}\int\_{t\_{0}}^{t\_{N}}\ell\left(% X\_{t}^{u},u,t\right)dt+h\left(X\_{t\_{N}}^{u}\right),~{}~{}~{}X\_{t\_{0}}^{u}={% \mathbf{x}}\_{0},$ |  | (5) |
| --- | --- | --- | --- |

which satisfies a Partial Differential Equation (PDE) given below in Theorem A.1.



###### Theorem A.1 (HJB Equation, [fleming2012deterministic, basar2020lecture]).


If $V^{\*}$ has continuous partial derivatives, then it must satisfy the following PDE, also known as Hamilton-Jacobi-Bellman (HJB) equation:



|  | $\displaystyle-\frac{\partial V^{\*}}{\partial t}\left({\mathbf{x}},t\right)=% \min\_{u\in{\mathcal{U}}}\left[H\left({\mathbf{x}},\nabla\_{\mathbf{x}}V^{\*}% \left({\mathbf{x}},t\right),u,t\right)\coloneqq\ell\left({\mathbf{x}},u,t% \right)+\left(\nabla\_{\mathbf{x}}V^{\*}\left({\mathbf{x}},t\right)\right)^{T}v% \left({\mathbf{x}},u,t\right)\right].$ |  |
| --- | --- | --- |

Also, the Hamiltonian $H\left({\mathbf{x}},\nabla\_{\mathbf{x}}V^{\*}\left({\mathbf{x}},t\right),u,t\right)$, optimal controller $u^{\*}(t)$ and the state trajectory ${\mathbf{x}}^{\*}(t)$ must satisfy



|  | $\displaystyle\min\_{u\in{\mathcal{U}}}H\left({\mathbf{x}}^{\*}(t),\nabla\_{% \mathbf{x}}V^{\*}\left({\mathbf{x}}^{\*}(t),t\right),u,t\right)=H\left({\mathbf{% x}}^{\*}(t),\nabla\_{\mathbf{x}}V^{\*}\left({\mathbf{x}}^{\*}(t),t\right),u^{\*}(t)% ,t\right).$ |  |
| --- | --- | --- |



### A.1 Interpreting reverse-SDE as a solution to optimal control


For clarity, we restate the problem setup here and describe the main ideas from §4 in more details.
Problem setup:
We discuss a standard approach to derive the optimal controller in a special case of our control problem (2).
We substitute $t\leftarrow 1-t$ to account for the time reversal in the reverse-SDE (1).
In this setup, $X^{u}\_{0}\sim{\mathcal{N}}\left(0,\mathrm{I}\_{d}\right)$ and $X^{u}\_{1}\sim p\_{data}$.
We consider the following dynamic without the Brownian motion:



|  | $\displaystyle\mathrm{d}X\_{t}^{u}=v(X\_{t}^{u},u,t)\mathrm{d}t,~{}~{}~{}X\_{t\_{0}% }^{u}={\mathbf{x}}\_{0},$ |  | (6) |
| --- | --- | --- | --- |

where $0\leq t\_{0}\leq t\leq t\_{N}\leq 1$ and $v:\mathbb{R}^{d}\times\mathbb{R}^{d}\times[t\_{0},t\_{N}]\rightarrow\mathbb{R}^{d}$ denotes the drift field.
The optimal controller $u^{\*}$ can be derived by solving the Hamilton-Jacobi-Bellman (HJB) equation [fleming2012deterministic, basar2020lecture],
see Appendix A for details.
By certainty equivalence, the same $u^{\*}$ applies to a more general case with the Brownian motion [bridge], where



|  | $\displaystyle\mathrm{d}X\_{t}^{u}=v(X\_{t}^{u},u,t)\mathrm{d}t+\mathrm{d}W\_{t},~% {}~{}~{}X\_{t\_{0}}^{u}={\mathbf{x}}\_{0}.$ |  | (7) |
| --- | --- | --- | --- |

Therefore, without loss of generality, we analyze the reverse dynamic in the absence of the Brownian motion, and employ the same controller in more general cases with the Brownian motion.



Below, we consider a dynamical system whose drift field is chosen to minimize a transient trajectory cost and a terminal cost (weighted by $\gamma$) that enforces “closeness” to reference content $x\_{1}$. Proposition A.2 provides the structure of the optimal control
in the limiting setting where $\gamma\rightarrow\infty$. Furthermore, suppose we replace $x\_{1}$ with its conditional expectation (discussed in Remark A.3), the resulting dynamic, interestingly, is the standard reverse-SDE for the Orstein-Uhlenbeck (OU) diffusion process. This connection between optimal control (more precisely, classic Linear Quadratic Control) and the standard reverse-SDE provides us a path to study other diffusion problems (*e.g*. personalization \citepdreambooth,stylealigned,styledrop,instantstyle, image editing or inversion \citepnti,indi,psld,stsl,rout2023theoretical) through the lens of stochastic optimal control.



###### Proposition A.2 (Linear optimal control with quadratic cost \citepbridge).


Consider the control problem:



|  |  | $\displaystyle\min\_{u\in{\mathcal{U}}}\int\_{t\_{0}}^{1}\frac{1}{2}\left\|u(X^{u}% \_{t},t)\right\|^{2}dt+\frac{\gamma}{2}\left\|X^{u}\_{1}-x\_{1}\right\|\_{2}^{2},$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\text{where }\mathrm{d}X^{u}\_{t}=u(X^{u}\_{t},t)\,\mathrm{d}t,~{}~% {}~{}~{}~{}X^{u}\_{t\_{0}}=x\_{0}$ |  |
| --- | --- | --- | --- |

Then, in the limit when $\gamma\rightarrow\infty$, the optimal controller is given by $u^{\*}=\frac{x\_{1}-X^{u}\_{t}}{1-t}$, which yields $\mathrm{d}X^{u}\_{t}=\frac{x\_{1}-X^{u}\_{t}}{1-t}\mathrm{d}t$ for the deterministic case and $\mathrm{d}X^{u}\_{t}=\frac{x\_{1}-X^{u}\_{t}}{1-t}\mathrm{d}t+\mathrm{d}W\_{t}$ for the stochastic case.



The optimal controller for the problem presented in Proposition A.2 can be derived using established techniques from control theory \citepfleming2012deterministic, basar2020lecture, kappen2008stochastic; the specific form of the above result follows from \citepbridge (but without their momentum term). The key steps in this derivation include: (1) computing the Hamiltonian, (2) applying the minimum principle theorem to derive a set of differential equations, and (3) taking the limit as $\gamma\rightarrow\infty$.
These three steps are fundamental in deriving a closed-form solution.
The final step is critical for satisfying hard terminal constraint and is essential for the practical implementation of Algorithm 1 and Algorithm 2, as detailed in §4.



For generative modeling, the controlled dynamics described in Proposition A.2 cannot be directly applied.
This limitation arises because the optimal control $u^{\*}$ depends on the terminal state $x\_{1}$, making it non-causal or reliant on future information.
Inspired by recent advancements in flow-based generative models \citeplipman2022flow, rectflow, we make the optimal controller causal by replacing the terminal state with its conditional expectation given the current state, i.e., , *i.e*. $x\_{1}\leftarrow\mathbb{E}[X^{u}\_{1}|X^{u}\_{t}={\mathbf{x}}\_{t}]$.
This modification results in a controlled dynamic that can be simulated to produce a generative model incorporating principles from optimal control, as elaborated in Remark A.3.



###### Remark A.3 (Connections between diffusion-based generative modeling and stochastic optimal control).


Following conditional diffusion models and optimal transport paths \citeplipman2022flow,rectflow, where $X^{f}\_{t}=tX^{f}\_{0}+(1-t)\epsilon$, the state variable $X^{u}\_{t}$ is equal in distribution to $X^{f}\_{1-t}=(1-t)X^{f}\_{0}+t\epsilon,\,\epsilon\sim{\mathcal{N}}\left(0,%
\mathrm{I}\_{d}\right)$ after time reversal.
Now, we use Tweedie’s formula [efron2011tweedie] to compute the posterior mean:



|  | $\displaystyle\mathbb{E}\Big{[}X^{u}\_{1}|X^{u}\_{t}\Big{]}=\frac{X^{u}\_{t}}{1-t}% +\frac{t^{2}}{1-t}\nabla\log p\left(X^{u}\_{t},1-t\right).$ |  | (8) |
| --- | --- | --- | --- |

Substituting the posterior mean in the controlled reverse dynamic of Proposition A.2, we arrive at



|  | $\displaystyle\mathrm{d}X^{u}\_{t}$ | $\displaystyle=\frac{\left(\mathbb{E}\Big{[}X^{u}\_{1}|X^{u}\_{t}\Big{]}-X^{u}\_{t% }\right)}{\left(1-t\right)}\mathrm{d}t+\mathrm{d}W\_{t}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle=\Bigg{[}\frac{t}{(1-t)^{2}}X^{u}\_{t}+\frac{t^{2}}{(1-t)^{2}}% \nabla\log p(X^{u}\_{t},1-t)\Bigg{]}\mathrm{d}t+\mathrm{d}W\_{t}.$ |  |
| --- | --- | --- | --- |



We observe that the above equation is structurally the same as reverse-SDE associated with a forward Orstein-Uhlenbeck (OU) diffusion process. This relation between diffusion-based generative models and optimal control is further explored in the Appendices below.



Indeed, diffusion models \citepddpm, songscore, ldm, sdxl, sc provide an effective approximation to the terminal state of a denoising process. This approximation has been used for a variety of generative modeling tasks.
Also, the terminal state can be approximated using Tweedie’s formula \citepefron2011tweedie with a learned score function \citepddpm
111Alternatively, when the reverse process is described by a probability flow ODE, a trained neural network can directly predict the terminal state \citepddim..
By utilizing these pre-trained diffusion models, we can employ the connection to optimal control as discussed above to develop practically implementable generative models that incorporates terminal objectives such as style and personalization. Consequently, the subsequent sections are dedicated to deriving the optimal controller assuming a known terminal state; we will approximate this in practice using Tweedie’s formula as above.



### A.2 Incorporating personalized style constraints through a terminal cost


In this section, we derive the optimal controller when we have access to the reference style features $y\_{1}$ at the terminal time (instead of the content of the image encoded through $x\_{1}$).



###### Proposition A.4.


Suppose $A\in\mathbb{R}^{k\times d}$ be a linear style extractor that operates on the terminal state $X^{u}\_{1}\in\mathbb{R}^{d}$.
Given reference style features $y\_{1}$, consider the control problem:



|  | $\displaystyle\min\_{u\in{\mathcal{U}}}\int\_{t\_{0}}^{1}\frac{1}{2}\left\|u(X^{u}% \_{t},t)\right\|^{2}dt+\frac{\gamma}{2}\left\|AX^{u}\_{1}-y\_{1}\right\|\_{2}^{2},$ |  | (9) |
| --- | --- | --- | --- |
|  | $\displaystyle\text{where }\mathrm{d}X^{u}\_{t}=u(X^{u}\_{t},t)\,\mathrm{d}t,~{}~% {}~{}~{}~{}X^{u}\_{t\_{0}}=x\_{0},$ |  | (10) |
| --- | --- | --- | --- |

Then, in the limit when $\gamma\rightarrow\infty$, the optimal controller $u^{\*}=\frac{\left(A^{T}A\right)^{-1}A^{T}\left(y\_{1}-AX^{u}\_{t}\right)}{1-t}$, which yields the following controlled dynamic:



|  | $\displaystyle\mathrm{d}X^{u}\_{t}=\frac{\left(A^{T}A\right)^{-1}A^{T}\left(y\_{1% }-AX^{u}\_{t}\right)}{1-t}\mathrm{d}t.$ |  | (11) |
| --- | --- | --- | --- |



###### Proof.


We derive the closed-form solution of the optimal controller given a fixed terminal state condition.
This is similar to [bridge], where the reverse process is accelerated using momentum (see also [kappen2008stochastic, basar2020lecture] for further details on this approach). The distinction, however, lies in the treatment of the terminal constraint.
For completeness, we provide full details of the proof below.



To derive the closed-form solution222With slight abuse of notation, we use ${\mathbf{x}}\_{t}$ to denote $X^{u}\_{t}$ and ${\mathbf{u}}\_{t}$ to denote $u(X^{u}\_{t},t)$ in the deterministic case., recall from equation (5) that
$\ell({\mathbf{x}}\_{t},{\mathbf{u}}\_{t},t)=\frac{1}{2}\left\|{\mathbf{u}}\_{t}%
\right\|^{2}$ and the terminal cost $h({\mathbf{x}}\_{1})=\frac{\gamma}{2}\left\|A{\mathbf{x}}\_{1}-y\_{1}\right\|^{2}$.
Let ${\mathbf{p}}\_{t}$ represent $\nabla\_{\mathbf{x}}V^{\*}({\mathbf{x}},t)$ in Theorem A.1.
Then, the Hamiltonian of the control problem (9) is given by



|  | $\displaystyle H({\mathbf{x}}\_{t},{\mathbf{p}}\_{t},{\mathbf{u}}\_{t},t)$ | $\displaystyle=\ell({\mathbf{x}}\_{t},{\mathbf{u}}\_{t},t)+{\mathbf{p}}\_{t}^{T}{% \mathbf{u}}\_{t}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle=\frac{1}{2}\left\|{\mathbf{u}}\_{t}\right\|^{2}+{\mathbf{p}}\_{t}^% {T}{\mathbf{u}}\_{t}.$ |  |
| --- | --- | --- | --- |

Since the minimizer of the Hamiltonian is ${\mathbf{u}}\_{t}^{\*}=-{\mathbf{p}}\_{t}$, the value function becomes



|  | $\displaystyle V^{\*}=\min\_{{\mathbf{u}}\_{t}}H({\mathbf{u}}\_{t},{\mathbf{p}}\_{t}% ,{\mathbf{u}}\_{t},t)=H({\mathbf{u}}\_{t},{\mathbf{p}}\_{t},{\mathbf{u}}\_{t}^{\*},% t)=-\frac{1}{2}\left\|{\mathbf{p}}\_{t}\right\|^{2}.$ |  | (12) |
| --- | --- | --- | --- |

Now, we use minimum principle theorem [basar2020lecture] to obtain the following set of differential equations:



|  | $\displaystyle\frac{\mathrm{d}{\mathbf{x}}\_{t}}{\mathrm{d}t}$ | $\displaystyle=\nabla\_{{\mathbf{p}}}H\left({\mathbf{x}}\_{t},{\mathbf{p}}\_{t},{% \mathbf{u}}\_{t}^{\*},t\right)=-{\mathbf{p}}\_{t};$ |  | (13) |
| --- | --- | --- | --- | --- |
|  | $\displaystyle\frac{\mathrm{d}{\mathbf{p}}\_{t}}{\mathrm{d}t}$ | $\displaystyle=-\nabla\_{{\mathbf{x}}}H\left({\mathbf{x}}\_{t},{\mathbf{p}}\_{t},{% \mathbf{u}}\_{t}^{\*},t\right)=0;$ |  | (14) |
| --- | --- | --- | --- | --- |
|  | $\displaystyle{\mathbf{x}}\_{t\_{0}}$ | $\displaystyle=x\_{0};$ |  | (15) |
| --- | --- | --- | --- | --- |
|  | $\displaystyle{\mathbf{p}}\_{t\_{N}}$ | $\displaystyle=\nabla\_{{\mathbf{x}}}h\left({\mathbf{x}}\_{t\_{N}},t\_{N}\right)=% \gamma A^{T}\left(A{\mathbf{x}}\_{t\_{N}}-y\_{1}\right).$ |  | (16) |
| --- | --- | --- | --- | --- |

Integrating both sides of (13), we have



|  | $\displaystyle\int\_{t\_{0}}^{1}\mathrm{d}{\mathbf{x}}\_{t}=-\int\_{t\_{0}}^{1}{% \mathbf{p}}\_{t}\mathrm{d}t=-{\mathbf{p}}\left(1-t\_{0}\right),$ |  | (17) |
| --- | --- | --- | --- |

where the last equality is due to (14), which states that ${\mathbf{p}}\_{t}$ is a constant independent of time $t$.
This implies ${\mathbf{x}}\_{1}={\mathbf{x}}\_{t\_{0}}-{\mathbf{p}}(1-t\_{0})$. From (16), we know for $t\_{N}=1$ that



|  | $\displaystyle{\mathbf{p}}\_{1}$ | $\displaystyle=\gamma A^{T}\left(A{\mathbf{x}}\_{1}-y\_{1}\right)$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle=\gamma\left(A^{T}A\left(x\_{0}-{\mathbf{p}}(1-t\_{0})\right)-A^{T}% y\_{1}\right)$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle=\gamma A^{T}Ax\_{0}-\gamma A^{T}A{\mathbf{p}}\_{1}(1-t\_{0})-\gamma A% ^{T}y\_{1}$ |  | (18) |
| --- | --- | --- | --- | --- |

Rearranging (18) and solving for ${\mathbf{p}}\_{1}$, we get



|  | $\displaystyle{\mathbf{p}}\_{1}$ | $\displaystyle=\gamma\left(I+\gamma A^{T}A\left(1-t\_{0}\right)\right)^{-1}\left% (A^{T}Ax\_{0}-A^{T}y\_{1}\right)$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle=\left(\frac{I}{\gamma}+A^{T}A\left(1-t\_{0}\right)\right)^{-1}% \left(A^{T}Ax\_{0}-A^{T}y\_{1}\right)={\mathbf{p}}$ |  | (19) |
| --- | --- | --- | --- | --- |

Passing (19) through the limit $\gamma\rightarrow\infty$, we get



|  | $\displaystyle\lim\_{\gamma\rightarrow\infty}{\mathbf{p}}=\frac{\left(A^{T}A% \right)^{-1}\left(A^{T}Ax\_{0}-A^{T}y\_{1}\right)}{1-t\_{0}}.$ |  | (20) |
| --- | --- | --- | --- |

Therefore, the optimal control becomes
${\mathbf{u}}\_{t}^{\*}=-{\mathbf{p}}=-\frac{\left(A^{T}A\right)^{-1}\left(A^{T}A%
{\mathbf{x}}\_{t}-A^{T}y\_{1}\right)}{1-t}$, and the resulting dynamical system is given by



|  | $\displaystyle\mathrm{d}{\mathbf{x}}\_{t}=\frac{\left(A^{T}A\right)^{-1}A^{T}% \left(y\_{1}-A{\mathbf{x}}\_{t}\right)}{1-t}\mathrm{d}t,$ |  |
| --- | --- | --- |

for the deterministic process and



|  | $\displaystyle\mathrm{d}{\mathbf{x}}\_{t}=\frac{\left(A^{T}A\right)^{-1}A^{T}% \left(y\_{1}-A{\mathbf{x}}\_{t}\right)}{1-t}\mathrm{d}t+\mathrm{d}W\_{t},$ |  |
| --- | --- | --- |

for the stochastic process with the Brownian motion.
This completes the statement of the proof.
∎



Implications:
The optimal controller depends on the reference style features $y\_{1}$ at the terminal time (instead of the image content $x\_{1}$ as in Appendix A.1).
The reverse dynamic can be simulated in practice by using CSD \citepcsd as a style feature extractor and replacing $y\_{1}$ with the extracted style features from the expected terminal state $\mathbb{E}\Big{[}X^{u}\_{1}|X^{u}\_{t}\Big{]}$, as discussed in Remark A.3.
This makes the controller drift causal and non-anticipating future information.



### A.3 Incorporating personalized style constraint through modulation and a terminal cost


In this section, we study a control problem where the velocity field is a linear combination of the state and the control variable.
This problem is interesting to study because of the following reason.
The reverse-SDE dynamic of the standard OU process has a drift field of the form:



|  | $\displaystyle v\left(X\_{t},t\right)=-X\_{t}-2\nabla\log p(X\_{t},t).$ |  |
| --- | --- | --- |

For a Gaussian prior $X\_{0}\sim{\mathcal{N}}\left(0,\mathrm{I}\right)$, the law of the OU process satisfies $\nabla\log p\left(X\_{t},t\right)=-X\_{t}$, and the corresponding drift field becomes $v\left(X\_{t},t\right)=X\_{t}$. Our goal is to modulate this drift field using a controller $u\left(X^{u}\_{t},t\right)$. The result below provides the structure of the optimal control (again in the setting where the terminal objective is known; see Appendix A1).



###### Proposition A.5.


Suppose $A\in\mathbb{R}^{k\times d}$ be a linear style extractor that operates on the terminal state $X^{u}\_{1}\in\mathbb{R}^{d}$.
Let ${\mathbf{p}}\_{t}$ denote $\nabla\_{\mathbf{x}}V^{\*}({\mathbf{x}},t)$ in HJB equation (A.1).
Given reference style features $y\_{1}$, consider the control problem:



|  | $\displaystyle\min\_{u\in{\mathcal{U}}}\int\_{t\_{0}}^{1}\frac{1}{2}\left\|u(X^{u}% \_{t},t)\right\|^{2}dt+\frac{\gamma}{2}\left\|AX^{u}\_{1}-y\_{1}\right\|\_{2}^{2},$ |  | (21) |
| --- | --- | --- | --- |
|  | $\displaystyle\text{where }\mathrm{d}X^{u}\_{t}=\Big{[}X^{u}\_{t}+u(X^{u}\_{t},t)% \Big{]}\,\mathrm{d}t,~{}~{}~{}~{}~{}X^{u}\_{t\_{0}}=x\_{0},$ |  | (22) |
| --- | --- | --- | --- |

Then, the optimal controller becomes $u^{\*}(t)=-{\mathbf{p}}\_{t}$, where the instantaneous state $X^{u}\_{t}={\mathbf{x}}\_{t}$ and ${\mathbf{p}}\_{t}$ satisfy the following:



|  | $\displaystyle\begin{bmatrix}{\mathbf{x}}\_{t}\\ {\mathbf{p}}\_{t}\end{bmatrix}=\begin{bmatrix}x\_{0}e^{t}-\frac{\gamma}{2}A^{T}% \left(A{\mathbf{x}}\_{1}-y\_{1}\right)e^{1+t}+\frac{\gamma}{2}A^{T}\left(A{% \mathbf{x}}\_{1}-y\_{1}\right)e^{1-t}\\ \gamma A^{T}\left(A{\mathbf{x}}\_{1}-y\_{1}\right)e^{1-t}\end{bmatrix}.$ |  |
| --- | --- | --- |



###### Proof.


The proof of Proposition A.5 is similar to Proposition A.4.
One key distinction is the set of differential equations obtained using minimum principle theorem \citepbasar2020lecture.
We begin with the Hamiltonian:



|  | $\displaystyle H({\mathbf{x}}\_{t},{\mathbf{p}}\_{t},{\mathbf{u}}\_{t},t)$ | $\displaystyle=\ell({\mathbf{x}}\_{t},{\mathbf{u}}\_{t},t)+{\mathbf{p}}\_{t}^{T}% \left({\mathbf{u}}\_{t}+{\mathbf{x}}\_{t}\right)$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle=\frac{1}{2}\left\|{\mathbf{u}}\_{t}\right\|^{2}+{\mathbf{p}}\_{t}^% {T}{\mathbf{u}}\_{t}+{\mathbf{p}}\_{t}^{T}{\mathbf{x}}\_{t},$ |  |
| --- | --- | --- | --- |

which gives us the minimizer of the Hamiltonian ${\mathbf{u}}\_{t}^{\*}=-{\mathbf{p}}\_{t}$ and its value function becomes $V^{\*}=\min\_{{\mathbf{u}}\_{t}}H({\mathbf{u}}\_{t},{\mathbf{p}}\_{t},{\mathbf{u}}\_%
{t},t)=H({\mathbf{u}}\_{t},{\mathbf{p}}\_{t},{\mathbf{u}}\_{t}^{\*},t)=-\frac{1}{2%
}\lVert{\mathbf{p}}\_{t}\rVert^{2}+{\mathbf{p}}\_{t}^{T}{\mathbf{x}}\_{t}$.
By the minimum principle theorem \citepbasar2020lecture,



|  |  | $\displaystyle\dot{{\mathbf{x}}}\_{t}\coloneqq\frac{\mathrm{d}{\mathbf{x}}\_{t}}{% \mathrm{d}t}=\nabla\_{{\mathbf{p}}}H\left({\mathbf{x}}\_{t},{\mathbf{p}}\_{t},{% \mathbf{u}}\_{t}^{\*},t\right)=-{\mathbf{p}}\_{t}+{\mathbf{x}}\_{t};$ |  | (23) |
| --- | --- | --- | --- | --- |
|  |  | $\displaystyle\dot{{\mathbf{p}}}\_{t}\coloneqq\frac{\mathrm{d}{\mathbf{p}}\_{t}}{% \mathrm{d}t}=-\nabla\_{{\mathbf{x}}}H\left({\mathbf{x}}\_{t},{\mathbf{p}}\_{t},{% \mathbf{u}}\_{t}^{\*},t\right)=-{\mathbf{p}}\_{t};$ |  | (24) |
| --- | --- | --- | --- | --- |
|  |  | $\displaystyle{\mathbf{x}}\_{t\_{0}}=x\_{0};$ |  | (25) |
| --- | --- | --- | --- | --- |
|  |  | $\displaystyle{\mathbf{p}}\_{t\_{N}}=\nabla\_{{\mathbf{x}}}h\left({\mathbf{x}}\_{t\_% {N}},t\_{N}\right)=\gamma A^{T}\left(A{\mathbf{x}}\_{t\_{N}}-y\_{1}\right).$ |  | (26) |
| --- | --- | --- | --- | --- |

This leads to a coupled system of differential equations with boundary conditions as given below:



|  | $\displaystyle\begin{bmatrix}\dot{{\mathbf{x}}}\_{t}\\ \dot{{\mathbf{p}}}\_{t}\end{bmatrix}$ | $\displaystyle=\begin{bmatrix}1&-1\\ 0&-1\end{bmatrix}\begin{bmatrix}{\mathbf{x}}\_{t}\\ {\mathbf{p}}\_{t}\end{bmatrix};$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle{\mathbf{x}}\_{t\_{0}}$ | $\displaystyle=x\_{0};$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle{\mathbf{p}}\_{1}$ | $\displaystyle=\gamma A^{T}\left(A{\mathbf{x}}\_{1}-y\_{1}\right).$ |  |
| --- | --- | --- | --- |

This can be solved numerically using ODE solvers, see  [fleming2012deterministic, basar2020lecture] for details.
Denote $\dot{{\mathbf{q}}\_{t}}=\begin{bmatrix}\dot{{\mathbf{x}}}\_{t}\\
\dot{{\mathbf{p}}}\_{t}\end{bmatrix}$
and $\mathrm{M}=\begin{bmatrix}1&-1\\
0&-1\end{bmatrix}$.
We seek a solution of the form ${\mathbf{q}}(t)={\mathbf{q}}e^{\lambda t}$. If ${\mathbf{q}}(t)$ is a solution of the above problem, then it must satisfy the following eigen value problem:



|  | $\displaystyle{\mathbf{q}}e^{\lambda t}\lambda=\mathrm{M}{\mathbf{q}}e^{\lambda t}.$ |  | (27) |
| --- | --- | --- | --- |

Writing the characteristic polynomial of (27), we get $\det{\Big{(}\mathrm{M}-\lambda\mathrm{I}\Big{)}}=0$, which gives the eigen values $\lambda=\{1,-1\}$.
Substituting these eigen values, we have



|  | $\displaystyle\begin{bmatrix}0&-1\\ 0&-2\end{bmatrix}\begin{bmatrix}q\_{1}\\ q\_{2}\end{bmatrix}=\mathbf{0},~{}~{}~{}~{}\begin{bmatrix}2&-1\\ 0&0\end{bmatrix}\begin{bmatrix}q\_{1}\\ q\_{2}\end{bmatrix}=\mathbf{0},$ |  |
| --- | --- | --- |

which gives two fundamental solutions. By combining these two, we obtain the final solution



|  | $\displaystyle\begin{bmatrix}{\mathbf{x}}\_{t}\\ {\mathbf{p}}\_{t}\end{bmatrix}=\omega\begin{bmatrix}1\\ 0\end{bmatrix}e^{t}+\xi\begin{bmatrix}1\\ 2\end{bmatrix}e^{-t},$ |  |
| --- | --- | --- |

where $\omega$ and $\xi$ can be found using the boundary conditions.
Since ${\mathbf{x}}\_{0}=x\_{0}$ and ${\mathbf{p}}\_{1}=\gamma A^{T}\left(A{\mathbf{x}}\_{1}-y\_{1}\right)$, we get $\omega=x\_{0}-\frac{\gamma}{2}A^{T}\left(A{\mathbf{x}}\_{1}-y\_{1}\right)e$ and $\xi=\frac{\gamma}{2}A^{T}\left(A{\mathbf{x}}\_{1}-y\_{1}\right)e$.
Substituting the values of $\omega$ and $\xi$, we arrive at



|  | $\displaystyle\begin{bmatrix}{\mathbf{x}}\_{t}\\ {\mathbf{p}}\_{t}\end{bmatrix}=\begin{bmatrix}x\_{0}e^{t}-\frac{\gamma}{2}A^{T}% \left(A{\mathbf{x}}\_{1}-y\_{1}\right)e^{1+t}+\frac{\gamma}{2}A^{T}\left(A{% \mathbf{x}}\_{1}-y\_{1}\right)e^{1-t}\\ \gamma A^{T}\left(A{\mathbf{x}}\_{1}-y\_{1}\right)e^{1-t}\end{bmatrix}.$ |  |
| --- | --- | --- |



This completes the proof of the proposition.



∎



Summary:
Though Appendices A.1-A.3, we have seen the connection between optimal control and diffusion based generation with a personalized terminal constraint. The general strategy has been to derive the optimal controller with known terminal state, and then replace the terminal state in the controller with its estimate using Tweedie’s formula. While the controllers so far have an explicit form, in practice, the data distribution is not Gaussian, and thus, we do not have a closed-form expression for the drift of the controller.



This line of analysis, however, points to our method RB-Modulation. As discussed in §4, we incorporate a consistent style descriptor in our controller’s terminal cost and numerically evaluate the drift of the controller at each reverse time step either through back propagation through the score network, or an approximation based on proximal gradient updates.





Input: Diffusion time steps $T$,
reference style image ${\mathbf{z}}\_{0}$,


          
style descriptor $\Psi(\cdot)$,
score network $s(\cdot,\cdot;\theta)$


Tunable parameters: Stepsize $\eta$, optimization steps $M$, proximal strength $\lambda$



Output: Personalized latent $X^{u}\_{0}$

1
Initialize ${\mathbf{x}}\_{T}\leftarrow{\mathcal{N}}\left(0,\mathrm{I}\_{d}\right)$

2
Initialize controller $u\in\mathbb{R}^{d}$

3
for *$t=T$ to $1$* do

4      
Compute posterior mean $\mathbb{E}\Big{[}X^{u}\_{0}|X^{u}\_{t}={\mathbf{x}}\_{t}\Big{]}=\frac{{\mathbf{x}%
}\_{t}}{\sqrt{\bar{\alpha}\_{t}}}+\frac{(1-\bar{\alpha}\_{t})}{\sqrt{\bar{\alpha}%
\_{t}}}s\left({\mathbf{x}}\_{t},t;\theta\right)$

5      
Initialize optimization variable ${\mathbf{x}}\_{0}=0$

6      
for *$m=1$ to $M$* do

7            
Compute controller’s cost ${\mathcal{L}}({\mathbf{x}}\_{0})\coloneqq\lVert\Psi({\mathbf{z}}\_{0})-\Psi({%
\mathbf{x}}\_{0})\rVert^{2}\_{2}+\lambda\lVert{\mathbf{x}}\_{0}-\mathbb{E}\left[X%
^{u}\_{0}|X^{u}\_{t}={\mathbf{x}}\_{t}\right]\rVert\_{2}^{2}$

8            
Update optimization variable ${\mathbf{x}}\_{0}={\mathbf{x}}\_{0}-\eta\nabla\_{{\mathbf{x}}\_{0}}{\mathcal{L}}({%
\mathbf{x}}\_{0})$


9       end for

10      Compute previous state ${\mathbf{x}}\_{t-1}=\text{DDIM}({\mathbf{x}}\_{0},{\mathbf{x}}\_{t})$ [ddim]


11 end for

return $X^{u}\_{0}$



Algorithm 2 Reference-Based Modulation (RB-Modulation) for large-scale generative models


![](x6.png)

Figure 6: Qualitative results of different tunable hyperparameters: Improved style-prompt disentanglement are shown when increasing to our best configurations optimization step size $\eta=0.1$ and optimization steps $M=3$.





Appendix B Additional Experimental Evaluation
---------------------------------------------


### B.1 Implementation details


Baselines:
We demonstrate the applicability of our method RB-Modulation with StableCascade [sc] (released before April 2024).
To our best knowledge, RB-Modulation is the first framework that introduces new capabilities to StableCascade by incorporating SOC and AFA modules.
Since there are no existing training-free personalization baselines designed for StableCascade, we seek alternatives built on other comparable state-of-the-art models such as SDXL \citepsdxl and Muse \citepmuse333Note that StableCascade and SDXL have comparable performance in prompt alignment whereas StableCascade is more efficient due to a highly compressed semantic latent space \citepsc..



Among alternate training-free baselines, InstantStyle \citepinstantstyle does not directly apply to StableCascade because it requires feature injection into specific layers of an IP-Adapter, which is not available for StableCascade. Similarly, StyleAligned \citepstylealigned relies on DDIM inversion, which is currently applicable only to single-stage diffusion models. In contrast, StableCascade utilizes a two-stage diffusion process, making the application of standard DDIM inversion \citepddim infeasible.
We run the official source code for InstantStyle444https://github.com/InstantStyle/InstantStyle and StyleAligned555https://github.com/google/style-aligned.
In the absence of a style description, we use “image in style” for DDIM inversion in StyleAligned.
Following InstantStyle \citepinstantstyle, we also compare with IP-Adapter. We include the quantitative comparison in Table 6.1, and only compare qualitatively with stronger baselines in Figure 3.



For completeness, we also compare with training-based baselines: StyleDrop \citepstyledrop and ZipLoRA \citepziplora.
Since the official codebase for StyleDrop666https://github.com/aim-uofa/StyleDrop-PyTorch and ZipLoRA777https://github.com/mkshing/ziplora-pytorch are not publicly available, we use the third-party implementation and follow the training details in the corresponding papers. It takes  5 minutes for training StyleDrop for 1000 steps and  20 minutes for training each LoRA for ZipLoRA. We train each LoRA with only one reference image for both content and styles to make a fair comparison with other methods. Similarly, we train StyleDrop with only one reference image. When a style description is not provided, we follow the original paper \citepstyledrop and use “in a [v\*] style” instead.



Tunable parameters. Our method introduces only two hyper-parameters: stepsize $\eta$ and optimization steps $M$ in Algorithm 1. We use DDIM sampling with $\eta=0.1$ and $M=3$ for all the experiments.



Content-style composition. The prompt-guided content-style composition task introduces a new layer of complexity beyond stylization §6.1. This task necessitates the disentanglement of the text prompt, reference style image, and reference content image through additional conditioning. Such complexity poses significant challenges for DDIM inversion \citepddim and attention caching mechanisms \citepstylealigned due to the inherent dependencies on multiple reverse paths.



Our AFA module effectively addresses these challenges.
It manipulates transformer layers to easily incorporate these additional conditions.
The content information is integrated in a manner similar to the style information. Specifically, we use a pre-trained ViT-L/14 model to extract content features in the SOC framework and update the latent embeddings concurrently via the AFA module, using an additional set of keys and values illustrated in Figure 2.



Furthermore, to better preserve the identity of the foreground content, we extract the desired content using LangSAM888https://github.com/luca-medeiros/lang-segment-anything based on the content prompt. This step is optional but offers more user control when multiple subjects are present in the reference image.



### B.2 Implementation using large-scale diffusion models


The exact implementation of our control problem (3) is given in Algorithm 1, which follows from our theoretical insights.
In practice, our controller encounters a challenge when the generative model contains billions of parameters as in StableCascade [sc] due to back propagation through the score network, as discussed in §4.
Our strategy to overcome this practical challenge involves a proximal gradient update, given in Line 7-8 of Algorithm 2.
To accelerate the sampling process, we run a few steps ($M=3$) of gradient descent after initializing ${\mathbf{x}}\_{0}=\mathbb{E}\left[X^{u}\_{0}|X^{u}\_{t}={\mathbf{x}}\_{t}\right]$, resulting in only two hyperparameters to tune: stepsize $\eta$ and the number of optimization steps $M$.
Further, since the CSD model expects a clean image to extract style features, we apply the previewer model available in StableCascade on the terminal state before extracting style features.
After obtaining the final personalized latent using our Algorithm 1 and Algorithm 2, we follow the decoding process as per the inference pipeline of the adopted generative model.


![](x7.png)

Figure 7: Impact of style descriptions in the prompt:
(a) When style descriptions are provided, all methods yield better results.
(b) Without style descriptions (*e.g*., hard for users to describe in text), alternative methods could struggle to capture the intended style in the reference image.
Our method offers consistent stylization even without explicit style descriptions.




### B.3 Impact of hyperparameters on controlling style and content features


As detailed in §4 and the ablation study in §6.1, SOC helps disentangle the style and the prompt information by updating the drift field in the standard reverse-SDE.
We study the impact of the two hyperparameters present in Algorithm 1 and Algorithm 2 that enables this disentanglement, as shown in Figure 6.
We found better disentanglement when the step size $\eta=0.1$ and the number of optimization steps $M=3$.
However, increasing the step size further results in style image information leaking into the output (top row). Additionally, adding more optimization steps increases computational overhead without yielding much performance gain (bottom row).



### B.4 Style description in text prompts for better assimilation of unique styles


In addition to the quantitative analysis in §6.1, Figure 7 demonstrates that our method generates consistent stylized results with and without the style description.
In contrast, the alternatives fail to accurately follow the prompt when the style description is absent.
Although all results show noticeable improvement when the style description is provided, it is often challenging for users to describe styles in many real-world scenarios.
We believe our early results by RB-Modulation will pave the way for interesting future research along this direction.



We present additional qualitative results on stylization with (Figure 9) and without (Figure 10) style descriptions using StyleAligned dataset \citepstylealigned. Our results consistently align with the reference style and the prompt, while other methods encounter several issues: (1) difficulty in following prompt guidance, (2) information leakage from the style reference image, and (3) failure to achieve reasonable prompt/style alignment in the absence of style descriptions.


![](x8.png)

Figure 8: Comparison of different evaluation metrics: The StableCascade output is provided for reference because it doesn’t use the reference style image. The highest score for each metric is marked bold with underscore. We compare four metrics: ImageReward and CLIP-T score for prompt alignment, DINO and CLIP-I score for style alignment. The prompt for the top row is “A cat” and for the bottom row is “A piano”.




### B.5 Challenges of evaluation metrics in measuring style and content leakage


In §6, we discussed the limitations of metrics used in previous works \citepstyledrop, stylealigned, ziplora, such as DINO \citepdino and CLIP-I score \citepclip. To quantify these limitations, we use results from our ablation study shown in Figure 4.
As illustrated in Figure 8, DINO and CLIP-I scores are not well-suited for measuring style similarity in the presence of content leakage. This is because images with high semantic correlations to the reference style image consistently receive higher scores. For instance, in the top row, although the last two columns visually align more closely with the isometric illustration styles of the reference image, the DirectConcat output featuring a lighthouse receives higher scores. The margin is particularly pronounced for CLIP-I score.



A similar observation can be made in the bottom row, where images containing train-related objects receive higher scores regardless of their stylistic similarity. Conversely, images with less content leakage (as seen in the last column) are assigned lower scores. This indicates that DINO and CLIP-I scores prioritize semantic content over stylistic fidelity, thus failing to accurately measure style similarity in scenarios where content leakage prevails.



On the other hand, our final method (last column), which combines AFA and SOC, demonstrates high scores for both prompt alignment metrics: ImageReward \citepimagereward and CLIP-T \citepclip. This method also shows higher user preference, as evidenced in Table 6.1. In contrast, the DirectConcat results suffer from information leakage and poor alignment with the prompt, resulting in significantly lower or even negative reward scores.



In the ablation study, our primary focus is on the disentanglement of prompts and reference styles. The conventional metrics fail to accurately reflect true performance due to information leakage. Consequently, we emphasize qualitative demonstrations and place greater importance on user study results, as shown in Table 6.1, similar to previous approaches \citepstylealigned, styledrop.



### B.6 More qualitative results on stylization and content-style composition


We also showcase results on consistent style generation using user defined prompts in Figure 11. Our results with different prompts consistently align with the styles while introducing various scenarios following the prompts. The other methods face challenges like information leakage (*e.g*. hiking boots and the monocular) and monotonous scenes (*e.g*. InstantStyle).
Note that the original StyleDrop paper [styledrop] has mentioned its difficulty when training with one image without description. We keep the results for completeness even though they are less satisfying.
Besides, we also demonstrate more qualitative results for content-style composition in Figure 13.


![](x9.png)

Figure 9: Additional qualitative results for stylization with style description: While the alternative methods face challenges like following the prompts (*e.g*., multiple airplanes instead of an airplane) and information leakage (*e.g*., the clouds on the cornflake bowl and the guitar in the milkshake image), our method demonstrates strong performance on both prompt and style alignment.
Style description is in blue.



![](x10.png)

Figure 10: Additional qualitative results for stylization without style description: StyleAligned and StyleDrop show severe performance drop after removing the style descriptions (*e.g*., see fireman and cat images). InstantStyle results show more information leakage (*e.g*., the pink ladybug and leopard), whereas no obvious performance drop is observed in our results.



![](x11.png)

Figure 11: Additional qualitative results for consistent stylization for user defined prompts: With no style description, our results demonstrate more diversity while following the styles and prompts. InstantStyle results show monotonous scenes and StyleAligned results suffer from severe information leakage. We report StyleDrop results for completeness and it is known to perform worse with no style description and single training image [styledrop].



![](x12.png)

Figure 12: User study interface: Three randomly sampled outputs are shown for each method given a style reference image, forming two rows of images. The users are asked to answer three questions on (1) style alignment (2) prompt alignment and (3) overall alignment and quality.



![](x13.png)

Figure 13: Additional qualitative results for content-style composition: Our results show better prompt and style alignment while preserving reference content without leaking contents from the reference style images (*e.g*. background of the first column and fruits in the last column,). Unlike compared baselines, our method is not restricted to a fixed pose of the reference content image, illustrating sample diversity.




### B.7 Human evaluation to discern highly subjective nature of style


We conduct a user study with 155 participants via Amazon Mechanical Turk using 100 styles from the StyleAligned dataset [stylealigned].
The study requires no personally identifiable information of the participants.
There is no risk incurred and no vulnerable population.
The standard guidelines have been followed while conducting the user study.



We first provide participants with instructions to familiarize them with the relevant terminologies.
For each style, we randomly sample three outputs using three different prompts.
Participants see two rows of model outputs in random order (3 images per row) and answer 3 questions, as illustrated in Figure 12.

1. 1.

In which row below, the images align better with the reference style image?
2. 2.

In which row below, the images align better with the reference text prompt above each image?
3. 3.

In which row below, the images overall align better with the reference style image AND the text prompt above each image AND with high quality?

For each question, participants choose one of three options. We collect 8 responses for each question, with each question comparing our method against one of the alternatives. In total, we gathered 7,200 responses.



### B.8 Failure cases of training-free stylization using RB-Modulation


In Figure 14, we illustrate stylization of different letters using a single reference style image.
Although our method captures the intended style and generates prompted letters, we notice that there is an inherent tendency to generate upper-case letters (Figure 14 (a)), even though it is prompted to generate lower-case letters.
Upon further investigation, we observed that this issue stems from the underlying generative model StableCascade, as shown in Figure 14 (b).
This highlights a crucial limitation of our method.
As a training-free method, RB-Modulation shares a concern with other training-free methods \citepinstantstyle,stylealigned,ssa that the performance is influenced by the original generative prior.


![](x14.png)

Figure 14: Failure cases for stylization:
The top row shows the results of our method, RB-Modulation, while the bottom row displays the results of the backbone, StableCascade. Notably, the stylized images do not adhere to the prompt,“lower-case letter”. This highlights the limitations imposed by the pre-trained generative priors on the capabilities of training-free personalization models (top row).





Appendix C Broader impact statement
-----------------------------------


Social impact: Image stylization and content-style composition based on diffusion models potentially have both positive and negative social impact. This technology provides an easy-to-use tool to the general public for image generation which can help visualize their artistic ideas.
On the other hand, our work on stylization and content-style composition poses a risk of generating arts that closely mimic or infringe upon existing copyrighted material, leading to legal and ethical issues. More broadly, our method inherits the risks from T2I models which are capable of generating fake contents that can be misused by malicious users.



Safeguards: We build on StableCascade \citepsc, which has a mechanism to filter offensive image generations. Since our method RB-Modulation builds on this pre-trained generative model, we inherit these safeguards.