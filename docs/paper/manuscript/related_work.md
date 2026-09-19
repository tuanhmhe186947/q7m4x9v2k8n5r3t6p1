# 2. Related Work

## 2.1 Multi-Object Tracking in Precision Livestock Farming

Automated visual tracking of individual animals is a cornerstone of precision livestock farming.
Early approaches in swine tracking relied heavily on classical computer vision techniques, such
as
adaptive background subtraction, frame differencing, and optical flow combined with
morphological
filtering or ellipse fitting. While effective under low stocking densities and controlled
lighting,
classical methods degrade rapidly in commercial environments characterized by shifting
illumination,
soiled pens, and physical crowding.

The advent of deep learning transformed animal multi-object tracking through detection-based
tracking
paradigms. Standard two-stage and one-stage object detectors (Faster R-CNN, SSD, YOLO families)
have
been paired with association algorithms such as SORT, DeepSORT, ByteTrack, BoT-SORT, and OC-SORT
to
track pigs, cattle, and poultry. In swine monitoring specifically, ByteTrack has gained
prominence
due to its association strategy that matches low-confidence detection boxes to retain tracks
through
partial occlusions.

However, group-housed pigs present tracking difficulties rarely encountered in standard human
MOT
benchmarks (e.g., MOT17, MOT20). Pigs housed under commercial conditions share uniform pinkish
or
white pigmentation, lack distinct visual markings, exhibit highly deformable postures (standing,
lying laterally, sternal recumbency, huddling), and frequently crowd tightly at communal feeding
and
drinking troughs. Visual re-identification (Re-ID) embeddings trained on appearance features
frequently confuse penmates because inter-individual visual variance is often smaller than
intra-individual posture variance. Furthermore, existing studies typically evaluate tracking on
short
video clips (10 to 60 seconds) and report aggregate frame-level metrics such as MOTA, HOTA, and
IDF1,
leaving the longitudinal identity stability of tracking pipelines unexamined.

## 2.2 Vision-Based Pig Behavior Recognition

Automated quantification of swine behavior has progressed from wearable sensor-based
technologies to
contactless computer vision. While wearable sensors (ear tags, collars, accelerometers, RFID)
provide
direct kinematic or proximity data, they are invasive, prone to detachment, mechanically damaged
by
chewing, and costly to scale to thousands of market pigs. Consequently, camera-based visual
monitoring has emerged as the preferred non-intrusive paradigm.

Computer vision approaches to pig behavior recognition have evolved from single-frame 2D CNNs to
spatio-temporal deep architectures. Single-frame classifiers analyze static posture (standing,
sitting, lying) but cannot resolve dynamic actions such as drinking, rooting, or fighting. To
model
temporal dynamics, researchers have employed 3D convolutional networks (C3D, I3D, SlowFast),
recurrent architectures (ConvLSTM), and spatio-temporal Transformers (TimeSformer, Video Vision
Transformers).

A persistent challenge in swine ethogram recognition is distinguishing visually similar
behaviors
that differ primarily in social context or fine kinematic nuances. For example, aggressive
fighting
and non-aggressive social-nose investigation both involve close head-to-head proximity between
two
pigs; however, fighting is marked by vigorous head knocking, biting, and rapid angular
displacement,
whereas social exploration is characterized by gentle sniffing and low-velocity contact.
Similarly,
feeding involves prolonged snout contact within a feed hopper, which visually resembles
exploratory
floor rooting near pen boundaries. Furthermore, commercial pig behavior datasets exhibit severe
class imbalance: maintenance behaviors like lying and standing dominate 70% to 90% of
observation
time, whereas high-impact welfare events like fighting or play comprise less than 2% of
observations.

## 2.3 Multimodal and Social Context Modeling

To overcome the visual ambiguity of isolated animal crops, recent literature emphasizes
multimodal
feature fusion and social context representation. In human action recognition, combining RGB
video
with spatial bounding-box geometry and optical flow or coordinate trajectories consistently
yields
higher discriminative capability. In animal behavior analysis, incorporating bounding-box
geometric
attributes (center coordinates, area, aspect ratio, orientation) provides vital spatial context
regarding an animal's location relative to stationary pen resources (feeders, drinkers,
enrichment
objects).

Moreover, because swine behavior is inherently social, an individual's behavioral state is
frequently influenced by or coordinated with penmates. Graph Convolutional Networks (GCNs) and
spatial-temporal graph attention networks have been explored to represent penmates as graph
nodes,
modeling inter-individual spatial distances and relative velocities as edge features. However,
fully
connected graph architectures often suffer from high computational complexity and over-smoothing
when scaled to dense commercial pens. This motivates the development of lightweight social
representations that capture partner behavior and proximity while preserving computational
efficiency.

## 2.4 Tracking Error Propagation and Downstream Behavioral Profiling

Despite substantial parallel progress in multi-object tracking and action recognition, the two
fields have developed largely in isolation. In the broader computer vision literature, tracking
benchmarks evaluate detection and association accuracy on raw bounding boxes, whereas action
recognition benchmarks evaluate classification on pre-segmented, identity-guaranteed video
clips.

In operational precision livestock farming, these two tasks are inextricably coupled: tracking
outputs serve as the spatial and temporal scaffolding for individual behavior recognition. When
an
upstream tracker switches the identities of two pigs, the downstream behavior classifier assigns
subsequent action predictions to the incorrect animal. If tracking identities switch during an
aggressive encounter, an aggressive pig may be labeled as victimized, or an inactive resting pig
may
be credited with excessive feeding bouts.

Prior literature in animal monitoring has occasionally acknowledged tracking drift as an
operational
concern, but systematic mathematical frameworks for quantifying this error propagation remain
lacking. Specifically, previous works have not isolated the exact functional relationship
between
upstream MOT association errors (such as IDSW and wrong-identity exposure duration) and
downstream
behavioral profile distortion under controlled, human-verified behavioral ground truth. This
work
specifically addresses this gap.
