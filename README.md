# ComParE 2022: MOS-C
## Mosquito Event Detection

A challenge to discover mosquitoes from the sounds of their wingbeats within real-world field data collected in South East Tanzania. Data is recorded in demanding conditions that include rural acoustic ambience, the presence of speech, rain, other insects, and many more acoustic phenomena.

* [Code submission instructions README](./docs/submission-instructions.md)
* [Baseline reproduction README](./docs/baseline-reproduction.md)
* [Zenodo data repository](https://zenodo.org/record/6478589)

### Context
In 2020 malaria caused around 241 million cases of disease across more than 100 countries resulting in an estimated 627,000 deaths
[[World Health Organization, 2022]](https://www.who.int/news-room/fact-sheets/detail/malaria). Mosquito surveys are used to establish vector species' composition and abundance, human biting
rates and thus the potential to transmit a pathogen. Traditional survey methods, such as human
landing catches, which collect mosquitoes as they land on the exposed skin of a collector, can be
time consuming, expensive, and are limited in the number of sites they can survey. These surveys can also expose collectors to disease.
Consequently, an affordable automated survey method
that detects, identifies and counts mosquitoes could generate unprecedented levels of high-quality
occurrence and abundance data over spatial and temporal scales currently difficult to achieve.

The HumBug project is a collaboration between the University of Oxford, the University of Surrey, and mosquito entomologists
worldwide [HumBug, 2021], which aims to to develop a mosquito acoustic sensor
that can be deployed into the homes of people in malaria-endemic areas to help monitor and identify
the mosquito species, allowing targeted and effective vector control. 

### Motivation

Your participation in this challenge can directly help to address our primary research question of detecting acoustic mosquito events for further downstream processing. The code has been ported from our deployed server, which has run the baseline model to successfully aid entomologists to study important properties of mosquito behaviour. Therefore, **any solutions you develop can be used directly in our deployment scenario, and your work will be appropriately credited for all subsequent research**. If you would like to read more about how we use this data and code, refer to [NeurIPS: HumBugDB: A large-scale acoustic mosquito dataset](https://arxiv.org/pdf/2110.07607.pdf), [ECML-PKDD: Automatic Acoustic Tagging with BNNs](https://link.springer.com/chapter/10.1007/978-3-030-86514-6_22).

### Data 
* `train` and `dev` available openly on: https://zenodo.org/record/6478589

Our data has been laboriously curated through 6+ years of data collection, which we have split into `train`, `dev`, and `test`. All of our `test` data is recorded in people's homes from various mosquito intervention solutions (e.g. HumBug bednets as detailed in [HumBugDB](https://arxiv.org/pdf/2110.07607.pdf)), which are used as a form of aid that can additionally provide us the opportunity to monitor and detect lethal mosquitoes from the sounds of their wingbeat. As the nature of the data is sensitive, this challenge _does not provide_ the `test` data directly, however, submissions are designed to evaluate over the test data. This allows preservation of data privacy, and removes the possibility of consciously or subconsciously tuning model design with knowledge of test data.

## Submission instructions
1. Sign [End-user License Agreement (EULA)](./docs/EULA_HumBugChallenge.pdf) and return to e-mail addresses as stated in the EULA.
2. Develop code according to instructions of [the submission template README](./docs/submission-instructions.md).
3. Run `make_submission.sh`
4. Upload resulting `submission.zip` to https://humbug.aml.mindfoundry.ai/.
    1. Use the password as supplied to the e-mail supplied in the EULA
    2. A total of five submissions are allowed per participant
    3. If your code has run into issues, we will notify you. Invalid submissions will not count towards the limit of 5.
5. Submit a paper with your methods, citing the papers as per the EULA, and the scores achieved on `dev` data with the local version of your code, to [TBA]
6. The scores on `test` will be revealed at the end of the challenge

## Support
Feel free to join our Slack channel on https://join.slack.com/t/compare-2022/shared_invite/zt-18jw9bstz-aYY6RmfuBR5WKZIzJxIQvA for help with setting up environments, or to clarify any potential issues. Alternatively, you may contact `ivankiskin1 AT gmail DOT com` directly.
