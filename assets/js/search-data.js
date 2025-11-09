// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-home",
    title: "home",
    section: "Navigation",
    handler: () => {
      window.location.href = "/2026/";
    },
  },{id: "nav-about",
          title: "about",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/2026/about/";
          },
        },{id: "nav-call-for-blogposts",
          title: "call for blogposts",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/2026/call/";
          },
        },{id: "nav-submitting",
          title: "submitting",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/2026/submitting/";
          },
        },{id: "nav-reviewing",
          title: "reviewing",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/2026/reviewing/";
          },
        },{id: "dropdown-lt-strong-gt-2026-lt-strong-gt",
              title: "&lt;strong&gt;2026&lt;/strong&gt;",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "https://iclr-blogposts.github.io/2026/";
              },
            },{id: "dropdown-2025",
              title: "2025",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "https://iclr-blogposts.github.io/2025/";
              },
            },{id: "dropdown-2024",
              title: "2024",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "https://iclr-blogposts.github.io/2024/";
              },
            },{id: "dropdown-2023",
              title: "2023",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "https://iclr-blogposts.github.io/2023/";
              },
            },{id: "dropdown-2022",
              title: "2022",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "https://iclr-blog-track.github.io/home/";
              },
            },{id: "post-the-thermodynamic-implications-of-gpu-cooling-systems-on-transformer-model-training-efficiency-during-leap-years",
        
          title: "The Thermodynamic Implications of GPU Cooling Systems on Transformer Model Training Efficiency During...",
        
        description: "A comprehensive analysis of how the additional day in leap years affects thermal dissipation patterns in datacenter GPU arrays, with particular emphasis on the correlation between Gregorian calendar anomalies and attention mechanism convergence rates.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/thermodynamic-gpu-cooling-leap-years/";
          
        },
      },{id: "post-attention-is-all-we-do-not-need",
        
          title: "Attention is All We Do Not Need",
        
        description: "A blogpost on why attention is actually garbage. The transformer revolution has spawned a cult-like devotion to attention mechanisms, with researchers mindlessly chanting attention is all you need like some sort of neural network mantra. This paper serves as an intervention, brutally exposing how attention mechanisms are nothing more than glorified lookup tables that have somehow convinced the entire field they&#39;re revolutionary. We demonstrate that these computationally bloated, interpretability-destroying monstrosities create more problems than they solve, inducing training instabilities that would make a Victorian-era scientist weep. Through extensive experiments, we show that literally any other architecture—carrier pigeons trained on matrix multiplication, abacuses running backpropagation, even a sufficiently motivated intern with a calculator—can outperform attention-based models while using fewer parameters than a smartphone calculator app. Our results suggest that the field&#39;s Stockholm syndrome relationship with attention has reached pathological levels, and we propose immediate therapeutic intervention through a return to architectures that don&#39;t require a small country&#39;s power grid to determine if a sentence is about cats or dogs.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/my-test/";
          
        },
      },{id: "post-research-directions-in-multimodal-chain-of-thought-mcot-with-sketching",
        
          title: "Research Directions in Multimodal Chain-of-Thought (MCoT) with Sketching",
        
        description: "This article explores adding sketching to Multimodal Chain-of-Thought (MCoT)reasoning to enhance AI capabilities. It reviews past methods, identifies key gaps such as the lack of sketch-rationale datasets, and proposes advancing the field through targeted data collection, unified multimodal models, and reinforcement learning. Ethical considerations include mitigating cultural bias and visual misrepresentation in generated sketches.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/mcot-sketching/";
          
        },
      },{id: "post-sample-blog-post",
        
          title: "Sample Blog Post",
        
        description: "Your blog post&#39;s abstract. Please add your abstract or summary here and not in the main body of your text. Do not include math/latex or hyperlinks.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/distill-example/";
          
        },
      },{id: "post-from-u-nets-to-dits-the-architectural-evolution-of-text-to-image-diffusion-models-2021-2025",
        
          title: "From U-Nets to DiTs: The Architectural Evolution of Text-to-Image Diffusion Models (2021–2025)",
        
        description: "A comprehensive analysis of how diffusion model architectures evolved from U-Net backbones to Diffusion Transformers, transforming text-to-image generation capabilities.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/diffusion-architecture-evolution/";
          
        },
      },{id: "post-defining-and-quantifying-compositional-structure",
        
          title: "Defining and quantifying compositional structure",
        
        description: "Compositionality is thought to be crucial in human cognition and AI, but we lack a scientific understanding of what it is. What kind of data is compositionally structured? Can we mathematically quanity the amount and character of compositional structure? This blog post introduces a novel approach for doing so, building off of existing tools from algorithmic information theory that formalize notions of complexity and structure. The mathematical definition of compositionality that we&#39;ll come to is rigorous, precise, and general, and the hope is that it can inspire novel research directions in AI for uncovering compositional structure in natural data.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/compositionality/";
          
        },
      },{id: "books-the-godfather",
          title: 'The Godfather',
          description: "",
          section: "Books",handler: () => {
              window.location.href = "/2026/books/the_godfather/";
            },},{id: "news-a-simple-inline-announcement",
          title: 'A simple inline announcement.',
          description: "",
          section: "News",},{id: "news-a-long-announcement-with-details",
          title: 'A long announcement with details',
          description: "",
          section: "News",handler: () => {
              window.location.href = "/2026/news/announcement_2/";
            },},{id: "news-a-simple-inline-announcement-with-markdown-emoji-sparkles-smile",
          title: 'A simple inline announcement with Markdown emoji! :sparkles: :smile:',
          description: "",
          section: "News",},{id: "projects-project-1",
          title: 'project 1',
          description: "with background image",
          section: "Projects",handler: () => {
              window.location.href = "/2026/projects/1_project/";
            },},{id: "projects-project-2",
          title: 'project 2',
          description: "a project with a background image and giscus comments",
          section: "Projects",handler: () => {
              window.location.href = "/2026/projects/2_project/";
            },},{id: "projects-project-3-with-very-long-name",
          title: 'project 3 with very long name',
          description: "a project that redirects to another website",
          section: "Projects",handler: () => {
              window.location.href = "/2026/projects/3_project/";
            },},{id: "projects-project-4",
          title: 'project 4',
          description: "another without an image",
          section: "Projects",handler: () => {
              window.location.href = "/2026/projects/4_project/";
            },},{id: "projects-project-5",
          title: 'project 5',
          description: "a project with a background image",
          section: "Projects",handler: () => {
              window.location.href = "/2026/projects/5_project/";
            },},{id: "projects-project-6",
          title: 'project 6',
          description: "a project with no image",
          section: "Projects",handler: () => {
              window.location.href = "/2026/projects/6_project/";
            },},{id: "projects-project-7",
          title: 'project 7',
          description: "with background image",
          section: "Projects",handler: () => {
              window.location.href = "/2026/projects/7_project/";
            },},{id: "projects-project-8",
          title: 'project 8',
          description: "an other project with a background image and giscus comments",
          section: "Projects",handler: () => {
              window.location.href = "/2026/projects/8_project/";
            },},{id: "projects-project-9",
          title: 'project 9',
          description: "another project with an image 🎉",
          section: "Projects",handler: () => {
              window.location.href = "/2026/projects/9_project/";
            },},{
        id: 'social-email',
        title: 'email',
        section: 'Socials',
        handler: () => {
          window.open("mailto:%79%6F%75@%65%78%61%6D%70%6C%65.%63%6F%6D", "_blank");
        },
      },{
        id: 'social-inspire',
        title: 'Inspire HEP',
        section: 'Socials',
        handler: () => {
          window.open("https://inspirehep.net/authors/1010907", "_blank");
        },
      },{
        id: 'social-rss',
        title: 'RSS Feed',
        section: 'Socials',
        handler: () => {
          window.open("/2026/feed.xml", "_blank");
        },
      },{
        id: 'social-scholar',
        title: 'Google Scholar',
        section: 'Socials',
        handler: () => {
          window.open("https://scholar.google.com/citations?user=qc6CJjYAAAAJ", "_blank");
        },
      },{
        id: 'social-custom_social',
        title: 'Custom_social',
        section: 'Socials',
        handler: () => {
          window.open("https://www.alberteinstein.com/", "_blank");
        },
      },{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];
