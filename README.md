<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/SkyjackerFWO/Mask_RCNN-Pytorch">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Best-README-Template</h3>

  <p align="center">
    An awesome README template to jumpstart your projects!
    <br />
    <a href="https://github.com/SkyjackerFWO/Mask_RCNN-Pytorch"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/SkyjackerFWO/Mask_RCNN-Pytorch">View Demo</a>
    ·
    <a href="https://github.com/SkyjackerFWO/Mask_RCNN-Pytorch/issues">Report Bug</a>
    ·
    <a href="https://github.com/SkyjackerFWO/Mask_RCNN-Pytorch/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

Mask-RCNN là mạng NN mở rộng từ Faster-RCNN. Năm 2020, Mask RCNN là SOTA (state of the art) cho bài toán về segmentation và object detection. Trên Github dã có rất nhiều repo code về Mask RCNN tuy nhiên hầu hết sử dụng Tensorflow hoặc sử dụng 1 bộ data kèm 1 ảnh mask.
Thế nên Repo này mình sẽ triển khai Mask RCNN với Pytorch và tập data được đánh theo chuẩn COCO.


<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

Ở đây tôi sử dụng 

* [![Pytorch][Pytorch.org]][Pytorch-url]
* [![Python][Python.org]][Python-url]
* [![Docker][Docker.com]][Docker-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

Trước khi chạy được, bạn phải có cài đạt Docker. Vì cơ bản Docker giúp tạo ra một môi trường tại chỗ tương đương với môi trường máy của tôi bằng 1 vài dòng ngắn.
Bạn có thể tham khảo blog này để biết docker-container cơ bản: <a href="https://aiot.phenikaa-uni.edu.vn/blog/nvidia-ngc-docker">Demo NVIDIA NGC</a>



### Installation

Khi đã cài đạt được Doker và nvidia-container-toolkit bạn hoàn toàn có thể bắt đầu 

1. Clone the repo
   ```sh
   git clone https://github.com/SkyjackerFWO/Mask_RCNN-Pytorch
   ```
2. CD to repo
   ```sh
   cd Mask_RCNN-Pytorch
   ```
3. Pull and match Docker (Tôi để memory swap là 10G nhưng mà thực tế ko cần nhiều đến thế, 5G chắc được. Nếu bạn gặp lỗi out of memory thì cân nhắc tăng lên)
   ```sh
   sudo docker run -memory-swap=10G --gpus all -it -d -v `pwd`:/codes/ duyanhskj/ocr-images:2.0
   ```
4. Run with docker
   ```sh
   cd ..
   cd codes
   python finetuning.py
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

Hiện tại thì code mới chỉ chạy cho vui thôi( vì tác giả lười chưa thêm lưu model. Việc này khá đơn giản nên mọi người cố đợi thêm thời gian nữa)

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

- [x] Push code and demo COCO data
- [ ] Thêm lưu model
- [ ] Tạo một container demo với Flask
- [ ] Hoàn thiện code 100%
- [ ] Hỗ trợ kiểu data
    - [ ] Coco
    - [ ] Mask image label

See the [open issues](https://github.com/SkyjackerFWO/Mask_RCNN-Pytorch/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>




<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

Use this space to list resources you find helpful and would like to give credit to. I've included a few of my favorites to kick things off!

*

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 