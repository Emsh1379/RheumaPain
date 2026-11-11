30 October 2026

The Editor-in-Chief\
*International Journal of Medical Informatics*

Dear Editor,

We submit the original research article **"Wrist-worn biosignals do not generalise for pain estimation in children with rheumatic disease: a leakage-aware benchmark and audit of a public dataset"** for consideration in the *International Journal of Medical Informatics*.

Wearable sensors are often proposed as an objective measure of pain, and studies frequently report accuracies near 90%. Using RheumaPain, the only public dataset of wrist-worn biosignals from children with rheumatic disease, we tested whether such signals estimate pain in children not seen during training. Our main findings are:

- The exercise/rest context alone predicts reported pain (AUC 0.71); no feature model or deep-learning model improved on it under leave-one-subject-out evaluation, and within a phase pain ranking was at chance.
- Random window splits inflate macro-F1 from 0.59 to 0.91 because 30-s windows identify the child with 90% accuracy; the high accuracy of subject-mixed evaluations therefore reflects leakage, not pain recognition.
- An audit of the public release found header rows stored as data, resampling distortion and a triplicated subject, which must be corrected before use.

We believe the work fits the journal's interest in the rigorous evaluation of health AI and in reproducible informatics methods. It offers practical guidance for wearable and clinical machine-learning studies: hold out whole participants, report context-only baselines, state the smallest detectable effect and audit public datasets. Reporting follows TRIPOD+AI, and the analysis code, cleaning rules, extracted features and all result tables are publicly available (link on the title page).

This manuscript is original, has not been published and is not under consideration elsewhere. Both authors approved the submission. The study is a secondary analysis of a public, de-identified dataset, received no specific funding, and the authors declare no competing interests. The use of generative AI is declared in the manuscript.

Thank you for considering our manuscript.

Sincerely,

Mohammad Emad Sharifi (corresponding author), on behalf of both authors\
Mother and Newborn Health Research Center, Shahid Sadoughi University of Medical Sciences, Yazd, Iran\
E-mail: emad.sharifi2001@gmail.com · ORCID: 0000-0003-2846-6964
