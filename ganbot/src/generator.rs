use anyhow::{Result, bail};

use tokio::{sync::{mpsc, oneshot}, spawn};
use url::Url;

const BASE_URL: &str = "https://brage.info/GAN/";
const BUFSIZ: usize = 8;

#[derive(Debug)]
pub struct GenResponse {
    pub final_url: Url,
    pub steps_url: Url,
}

struct GenRequest {
    prompt: String,
    quality: Quality,
    respond: oneshot::Sender<GenResponse>,
}

#[derive(Debug)]
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
        let base_url = Url::parse(BASE_URL).unwrap();

        spawn(async move {
            loop {
                let request = rx.recv().await.expect("Generator should never close!");
                println!("Generating {:?} in {:?} quality", request.prompt, request.quality);
                
                let result = "moo".to_string();
                let final_url = base_url.join(&result).expect("Should be valid URL!");
                let steps_url = base_url.join(&result).expect("Mooo!");
                let response = GenResponse{final_url, steps_url};

                request.respond.send(response).expect("Receiver should nexver hang up");
            }
        });
    }

    pub fn queue(&self, quality: Quality, prompt: String) -> Result<(usize, oneshot::Receiver<GenResponse>)> {
        let (tx, rx) = oneshot::channel();
        let request = GenRequest { prompt, quality, respond: tx };
        let qlen = BUFSIZ - self.tx.capacity();
        match self.tx.try_send(request) {
            Ok(()) => Ok((qlen + 1, rx)),
            Err(_) => bail!("Too many prompts already queued, try again later."),
        }
    } 
}