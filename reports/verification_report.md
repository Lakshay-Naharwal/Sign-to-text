# Phase 1 Verification: Signer Clustering & Split Structure

## 1. Visual Verification (Sample Frames)
As requested, here are representative frames sampled from the dataset confirming the stark difference between the 1080p (Train) and 240p (Val/Test) videos.

### Train Set: Student Volunteers (1080p)
Notice the landscape aspect ratio, higher resolution, and identical recording environment for the students.
````carousel
![1080p Sample 0 - bad (4).MP4](C:/Users/Asus/.gemini/antigravity-ide/brain/32750980-2294-47a0-ac43-dce61c9a4611/verification/1080p_samples/sample_0.jpg)
<!-- slide -->
![1080p Sample 1 - train (3).MP4](C:/Users/Asus/.gemini/antigravity-ide/brain/32750980-2294-47a0-ac43-dce61c9a4611/verification/1080p_samples/sample_1.jpg)
<!-- slide -->
![1080p Sample 2 - today (3).MP4](C:/Users/Asus/.gemini/antigravity-ide/brain/32750980-2294-47a0-ac43-dce61c9a4611/verification/1080p_samples/sample_2.jpg)
````

### Validation/Test Sets: Native Signers (240p)
Notice the portrait aspect ratio, lower resolution, and distinctly different physical environment.
````carousel
![240p Sample 0 - i am tired (2).mp4](C:/Users/Asus/.gemini/antigravity-ide/brain/32750980-2294-47a0-ac43-dce61c9a4611/verification/240p_samples/sample_0.jpg)
<!-- slide -->
![240p Sample 1 - i love you (2).mp4](C:/Users/Asus/.gemini/antigravity-ide/brain/32750980-2294-47a0-ac43-dce61c9a4611/verification/240p_samples/sample_1.jpg)
<!-- slide -->
![240p Sample 2 - i am fine. thank you sir (2).mp4](C:/Users/Asus/.gemini/antigravity-ide/brain/32750980-2294-47a0-ac43-dce61c9a4611/verification/240p_samples/sample_2.jpg)
````

## 2. Val/Test Signer Breakdown
The script successfully categorized the 195 native-signer videos. The breakdown is completely flawless, with **zero mixing** of signers between Validation and Test:

**Validation Set (99 videos total):**
- **Signer ID 5:** 99 videos
- *Signer ID 6:* 0 videos

**Test Set (96 videos total):**
- **Signer ID 6:** 96 videos
- *Signer ID 5:* 0 videos

This confirms that Native Signer 1 (ID 5) is perfectly isolated in Validation, and Native Signer 2 (ID 6) is perfectly isolated in Test.
