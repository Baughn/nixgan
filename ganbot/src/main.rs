use anyhow::{Result, bail, Context as _};
use std::env;
use serenity::{
    async_trait,
    model::{
        channel::Message,
        gateway::Ready,
        interactions::application_command::{
            ApplicationCommand,
            ApplicationCommandInteractionDataOptionValue,
            ApplicationCommandOptionType,
        },
        interactions::Interaction,
        interactions::InteractionResponseType,
    },
    prelude::*,
};
use tokio::spawn;
use tokio::time::{sleep, Duration};

mod generator;

struct Handler;

#[async_trait]
impl EventHandler for Handler {
    async fn interaction_create(&self, ctx: Context, interaction: Interaction) {
        if let Interaction::ApplicationCommand(command) = interaction {
            let response = command.create_interaction_response(&ctx.http, |r| {
                r
                    .kind(InteractionResponseType::ChannelMessageWithSource)
                    .interaction_response_data(|message| message.content("Testing..."))
            }).await;

            let origin = command.member.as_ref().map(|m| m.mention().to_string());

            spawn(async move {
                sleep(Duration::from_millis(1000)).await;
                let response = command.create_followup_message(&ctx.http, |r| {
                    let url = "https://brage.info/GAN/jaxgan/%2522Cosmic_cataclysm%2522_by_Dan_Mumford%2520%25281%2520of%25202%2529%2520at%252020220228025312.jpg";
                    r
                        .content(origin.unwrap_or("".into()))
                        .create_embed(|embed|
                            embed.description("description").image(url)
                        )
                }).await;
                if let Err(why) = response {
                    println!("Cannot respond to command: {}", why);
                }
            });

            if let Err(why) = response {
                println!("Cannot respond to command: {}", why);
            }
            
            // let content = match command.data.name.as_str() {
            //     "jaxgan" => {
            //         let options = command.data.options.get(0).expect("Expected prompt").resolved.as_ref().expect("Expected user object");
            //         if let ApplicationCommandInteractionDataOptionValue::String(prompt) = options {
            //             command.create_interaction_response(&ctx.http, |response| {
            //                 response
            //                     .kind(InteractionResponseType::ChannelMessageWithSource)
            //                     .interaction_response_data(|message| message;)
            //             });
            //             spawn(async move {
            //             });
            //             format!("Generating {}", prompt)
            //         } else {
            //             "Need a prompt!".into()
            //         }
            //     },
            //     _ => "not implemented :(".to_string(),
            // };

            // if let Err(why) = command
            //     .create_interaction_response(&ctx.http, |response| {
            //         response
            //             .kind(InteractionResponseType::ChannelMessageWithSource)
            //             .interaction_response_data(|message| message.content(content))
            //     })
            //     .await
            // {
            //     println!("Cannot respond to slash command: {}", why);
            // }
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
    // Configure the client with your Discord bot token in the environment.
    let token = env::var("DISCORD_TOKEN").expect("Expected a token in the environment");

    // The Application Id is usually the Bot User Id.
    let application_id: u64 = env::var("APPLICATION_ID")
        .context("Expected an application id in the environment")?
        .parse()
        .context("application id is not a valid id")?;

    // Build our client.
    let mut client = Client::builder(token)
        .event_handler(Handler)
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