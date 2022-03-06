use anyhow::{Result, bail};

use tokio::{sync::{mpsc, oneshot}, spawn, process::Command};
use url::Url;

const BASE_URL: &str = "https://brage.info/GAN/";
const BUFSIZ: usize = 8;

#[derive(Debug)]
pub struct GenResponse {
    pub message: Option<String>,
    pub final_url: Option<Url>,
    pub steps_url: Option<Url>,
}

struct GenRequest {
    prompt: String,
    quality: Quality,
    respond: oneshot::Sender<GenResponse>,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Quality {
    Normal, HQ
}

pub struct Generator {
    tx: mpsc::Sender<GenRequest>,
}

impl Generator {
    pub fn new() -> Generator {
        let (tx, rx) = mpsc::channel(BUFSIZ);
        spawn(async move {
            Generator::generator(rx)
        });
        Generator { tx }
    }

    fn generator(mut rx: mpsc::Receiver<GenRequest>) {
        spawn(async move {
            loop {
                let request = rx.recv().await.expect("Generator should never close!");
                let response = Generator::generate(&request).await;
                request.respond.send(
                    match response {
                        Ok(r) => r,
                        Err(why) => GenResponse { 
                            message: Some(why.to_string()),
                            final_url: None,
                            steps_url: None,
                        }
                    }
                ).expect("Discord channel should never close!");
            }
        });
    }

    async fn generate(request: &GenRequest) -> Result<GenResponse> {
        println!("Generating {:?} in {:?} quality", request.prompt, request.quality);

        let output = Command::new("nix")
            .current_dir("jax-diffusion")
            .arg("develop")
            .arg("-c")
            .arg("chrt")
            .arg("-b")
            .arg("0")
            .arg("python3")
            .arg("run.py")
            .arg(format!("{:?}", request.quality))
            .arg(request.prompt.clone())
            .output().await?;

        let base_url = Url::parse(BASE_URL).unwrap().join("jaxgan/").unwrap();
        let mut final_url = None;
        let mut steps_url = None;
        for line in String::from_utf8_lossy(&output.stdout).lines() {
            if let Some((_, image)) = line.split_once("IMAGE: ") {
                final_url = Some(base_url.join(image)?);
            }
            if let Some((_, dir)) = line.split_once("STEPS_DIR: ") {
                steps_url = Some(base_url.join(dir)?);
            }
        }

        println!("Finished generating (one way or the other");

        if final_url.is_none() {
            // Something went wrong. Report back what.
            let message = String::from_utf8_lossy(&output.stderr).into();
            println!("{}", message);
            Ok(GenResponse {
                message: Some(message),
                final_url,
                steps_url,
            })
        } else {
            Ok(GenResponse {
                message: None,
                final_url,
                steps_url,
            })
        }
    }

    pub fn queue(&self, quality: Quality, prompt: String) -> Result<(usize, oneshot::Receiver<GenResponse>)> {
        let (tx, rx) = oneshot::channel();
        let request = GenRequest { prompt, quality, respond: tx };
        if self.tx.is_closed() {
            panic!("Generator should never close!");
        }
        let qlen = BUFSIZ - self.tx.capacity();
        if quality == Quality::HQ && qlen > BUFSIZ / 2 && qlen < BUFSIZ {
            bail!("Too many prompts already queued, try normal quality.")
        }
        match self.tx.try_send(request) {
            Ok(()) => Ok((qlen + 1, rx)),
            Err(_) => bail!("Too many prompts already queued, try again later."),
        }
    } 
}