use bevy::diagnostic::{Diagnostics, FrameTimeDiagnosticsPlugin};
use bevy::prelude::*;

pub struct FpsPlugin;

impl Plugin for FpsPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugin(FrameTimeDiagnosticsPlugin)
            .add_startup_system(infotext_system)
            .add_system(change_text_system);
    }
}

#[derive(Component)]
struct TextChanges;

fn infotext_system(mut commands: Commands, asset_server: Res<AssetServer>) {
    let font = asset_server.load("fonts/FiraSans-Bold.ttf");

    commands
        .spawn_bundle(
            TextBundle::from_sections([
                TextSection::from_style(TextStyle {
                    font: font.clone(),
                    font_size: 30.0,
                    color: Color::ORANGE_RED,
                }),
                TextSection::new(
                    " fps, ",
                    TextStyle {
                        font: font.clone(),
                        font_size: 30.0,
                        color: Color::ORANGE_RED,
                    },
                ),
                TextSection::from_style(TextStyle {
                    font: font.clone(),
                    font_size: 30.0,
                    color: Color::GREEN,
                }),
                TextSection::new(
                    " ms/frame",
                    TextStyle {
                        font: font.clone(),
                        font_size: 30.0,
                        color: Color::BLUE,
                    },
                ),
            ])
            .with_style(Style {
                align_self: AlignSelf::FlexEnd,
                position_type: PositionType::Absolute,
                position: UiRect {
                    bottom: Val::Px(5.0),
                    right: Val::Px(15.0),
                    ..default()
                },
                ..default()
            }),
        )
        .insert(TextChanges);
}

fn change_text_system(
    time: Res<Time>,
    diagnostics: Res<Diagnostics>,
    mut query: Query<&mut Text, With<TextChanges>>,
) {
    for mut text in &mut query {
        let mut fps = 0.0;
        if let Some(fps_diagnostic) = diagnostics.get(FrameTimeDiagnosticsPlugin::FPS) {
            if let Some(fps_avg) = fps_diagnostic.average() {
                fps = fps_avg;
            }
        }

        let mut frame_time = time.delta_seconds_f64();
        if let Some(frame_time_diagnostic) = diagnostics.get(FrameTimeDiagnosticsPlugin::FRAME_TIME)
        {
            if let Some(frame_time_avg) = frame_time_diagnostic.average() {
                frame_time = frame_time_avg;
            }
        }
        text.sections[0].value = format!("{:.1}", fps);

        text.sections[2].value = format!("{:.3}", frame_time);
    }
}
