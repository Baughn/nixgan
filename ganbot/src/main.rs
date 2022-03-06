use anyhow::{Result, bail, Context as _};
use generator::{Generator, Quality};

use std::env::{self, set_current_dir};
use serenity::{
    async_trait,
    model::{
        gateway::Ready,
        interactions::application_command::{
            ApplicationCommand,
            ApplicationCommandInteractionDataOptionValue,
            ApplicationCommandOptionType,
        },
        interactions::{Interaction, application_command::ApplicationCommandInteraction},
        interactions::InteractionResponseType,
    },
    prelude::*,
};
use tokio::spawn;

mod generator;

const BASE_DIR: &str = "/home/svein/dev/nixgan/";

struct Handler {
    generator: Generator,
}

impl Handler {
    async fn generate(&self, quality: Quality, prompt: String, ctx: Context, command: ApplicationCommandInteraction, user: String) -> Result<usize> {
        let result = self.generator.queue(quality, prompt.clone())?;

        spawn(async move {
            let http = &ctx.http;
            let generator_response = result.1.await;
            // Some time later...
            match generator_response {
                Err(why) => println!("Unknown generation error: {}", why),
                Ok(generated_picture) => {
                    let description = prompt;
                    let text = if let Some(msg) = generated_picture.message {
                        format!("{} — Something went wrong: {}", user, msg)
                    } else if let Some(psteps) = generated_picture.steps_url {
                        format!("{} – earlier steps: {}", user, psteps)
                    } else {
                        format!("{} — but the earlier steps are mysteriously missing.", user)
                    };
                    let result = command.create_followup_message(http, |response| {
                        response
                            .content(text)
                            .create_embed(|embed| {
                                let embed = embed.description(description);
                                if let Some(image) = generated_picture.final_url {
                                    embed.image(image)
                                } else {
                                    embed
                                }
                            })
                    }).await;
                    if let Err(why) = result {
                        println!("Unknown error sending response: {}", why);
                    }
                }
            }
        });

        Ok(result.0)
    }
}

#[async_trait]
impl EventHandler for Handler {
    async fn interaction_create(&self, ctx: Context, interaction: Interaction) {
        let http = &ctx.http;
        if let Interaction::ApplicationCommand(command) = interaction {
            let response_text = {
                if let Some(user) = command.member.as_ref().map(|m| m.mention().to_string()) {
                    // Legal usage in this branch.
                    // Go ahead and generate. Maybe.
                    let quality = match command.data.name.as_ref() {
                        "jaxgan" => Quality::Normal,
                        "jaxganhq" => Quality::HQ,
                        _ => panic!("Unknown command!")
                    };
                    let prompt = command.data.options.get(0)
                        .expect("Extracting parameter")
                        .resolved.as_ref()
                        .expect("Expected parameter string");
                    if let ApplicationCommandInteractionDataOptionValue::String(prompt) = prompt {
                        match self.generate(quality, prompt.clone(), ctx.clone(), command.clone(), user).await {
                            Err(why) => why.to_string(),
                            Ok(qlen) => format!("Generating. You are #{} in the queue.\nPrompt was: {}", qlen, prompt),
                        }
                    } else {
                        "Unknown error extracting prompt.".to_string()
                    }
                } else {
                    "For private usage, please visit https://pharmapsychotic.com/tools.html and use a Colab notebook.".to_string()
                }
            };
            let result = command.create_interaction_response(http, |response| {
                response
                    .kind(InteractionResponseType::ChannelMessageWithSource)
                    .interaction_response_data(|m| m.content(response_text))
            }).await;
            if let Err(why) = result {
                println!("Error in interaction_create: {}", why);
            }
        }
    }

    async fn ready(&self, ctx: Context, ready: Ready) {
        println!("{} is connected!", ready.user.name);

        let commands = vec![
            ApplicationCommand::create_global_application_command(&ctx.http, |command| {
                command
                    .name("jaxgan").description("Generate a picture using JAX-GAN (standard quality)")
                    .create_option(|option| {
                        option
                            .name("prompt")
                            .description("The prompt to use")
                            .kind(ApplicationCommandOptionType::String)
                            .required(true)
                    })
            })
            .await,
            ApplicationCommand::create_global_application_command(&ctx.http, |command| {
                command
                    .name("jaxganhq").description("Generate a picture using JAX-GAN (high quality, slow)")
                    .create_option(|option| {
                        option
                            .name("prompt")
                            .description("The prompt to use")
                            .kind(ApplicationCommandOptionType::String)
                            .required(true)
                    })
            })
            .await,
        ];        

        println!("I created the following global slash commands: {:#?}", commands);
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Chdir to the expected cwd.
    set_current_dir(BASE_DIR).unwrap();

    // Configure the client with your Discord bot token in the environment.
    let token = env::var("DISCORD_TOKEN").expect("Expected a token in the environment");

    // The Application Id is usually the Bot User Id.
    let application_id: u64 = env::var("APPLICATION_ID")
        .context("Expected an application id in the environment")?
        .parse()
        .context("application id is not a valid id")?;

    // Build our client.
    let mut client = Client::builder(token)
        .event_handler(Handler { generator: Generator::new() })
        .application_id(application_id)
        .await
        .context("Error creating client")?;

    // Finally, start a single shard, and start listening to events.
    //
    // Shards will automatically attempt to reconnect, and will perform
    // exponential backoff until it reconnects.
    if let Err(why) = client.start().await {
        println!("Client error: {:?}", why);
    }

    bail!("GANBot is not expected to exit");
}