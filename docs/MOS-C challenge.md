# ComParE 2022: MOS-C
## Mosquito Event Detection

A challenge to discover mosquitoes from the sounds of their wingbeats within people's homes in South East Tanzania. Data is recorded in demanding real-world conditions such as rural acoustic scenes, the presence of speech, rain, other insects, and many more.

### Context
In 2019, for example, malaria caused around 229
million cases of disease across more than 100 countries resulting in an estimated 409,000 deaths
[World Health Organization, 2020]. Mosquito surveys are used to establish vector species' composition and abundance, human biting
rates and thus the potential to transmit a pathogen. Traditional survey methods, such as human
landing catches, which collect mosquitoes as they land on the exposed skin of a collector, can be
time consuming, expensive, and are limited in the number of sites they can survey. These surveys can also expose collectors to disease.
Consequently, an affordable automated survey method
that detects, identifies and counts mosquitoes could generate unprecedented levels of high-quality
occurrence and abundance data over spatial and temporal scales currently difficult to achieve.

The HumBug project is a collaboration between the University of Oxford and mosquito entomologists
worldwide [HumBug, 2021], which aims to to develop a mosquito acoustic sensor
that can be deployed into the homes of people in malaria-endemic areas to help monitor and identify
the mosquito species, allowing targeted and effective vector control. 

### Motivation

Your participation in this challenge can directly help to address our primary research question of detecting acoustic mosquito events for further downstream processing. The code has been ported from our deployed server, which has run the baseline model to successfully aid entomologists to study [xyz]. Therefore, **any solutions you develop can be used directly in our deployment scenario, and your work will be appropriately credited for all subsequent research**. If you would like to read more about how we use this data and code, refer to [] []

### Data
Our data has been laboriously crafted through 6+ years of data collection, which we have split into `train`, `dev`, and `test`. All of our `test` data is recorded in people's homes from various mosquito intervention solutions (e.g. HumBug bednets [ref]), which are used as a form of aid that can additionally provide us the opportunity to monitor and detect lethal mosquitoes from the sounds of their wingbeat. As the nature of the data is sensitive, this challenge _does not provide_ the `test` data directly, however, submissions are designed to evaluate over the test data.

## Submission instructions
1. Sign EULA [here]
2. Develop code according to instructions of [].
3. Run `make_submission.sh`
4. Upload resulting `submission.zip` to https://humbug.aml.mindfoundry.ai/.
    1. Use the username and password as supplied to the e-mail used in the EULA
    2. A total of five submissions are allowed per participant
    3. If your code has run into issues, we will notify you. Invalid submissions will not count towards the limit of 5.
5. Submit a paper with your methods, and the scores achieved on `dev` data with the local version of your code, to [...]
6. The scores on `test` will be revealed at the end of the challenge
