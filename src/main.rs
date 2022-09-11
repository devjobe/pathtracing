mod fps;
mod pathtrace_node;
use bevy::{
    core_pipeline::clear_color::ClearColorConfig,
    input::{keyboard::KeyboardInput, ButtonState},
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages},
    window::WindowResized,
};
use fps::FpsPlugin;
use pathtrace_node::{PathTracingImage, PathTracingUniform};

use crate::pathtrace_node::PathTraceNodePlugin;

fn main() {
    let mut app = App::new();
    app.insert_resource(WindowDescriptor {
        title: "Path tracing".to_string(),
        ..Default::default()
    })
    .add_plugins(DefaultPlugins)
    .add_plugin(PathTraceNodePlugin)
    .add_plugin(FpsPlugin)
    .add_startup_system(setup)
    .add_system(update)
    .add_system(update_image_to_window_size);
    app.run();
}

fn update(
    mut uniforms: ResMut<PathTracingUniform>,
    time: Res<Time>,
    mut timer: Local<f64>,
    mut keyboard_input_events: EventReader<KeyboardInput>,
    asset_server: Res<AssetServer>,
    mut is_loading: Local<bool>,
    keyboard: Res<Input<KeyCode>>,
    mut resize_events: EventReader<WindowResized>,
) {
    uniforms.frame = uniforms.frame + 1.0;

    let mut reset = false;
    for event in keyboard_input_events.iter() {
        if event.state == ButtonState::Pressed && event.key_code == Some(KeyCode::Space) {
            reset = true;
        }
    }

    for resize_event in resize_events.iter() {
        if resize_event.id.is_primary() {
            reset = true;
        }
    }

    {
        let x = keyboard.pressed(KeyCode::D) as i32 - keyboard.pressed(KeyCode::A) as i32;
        let y = keyboard.pressed(KeyCode::Q) as i32 - keyboard.pressed(KeyCode::E) as i32;
        let z = keyboard.pressed(KeyCode::W) as i32 - keyboard.pressed(KeyCode::S) as i32;

        let has_movement = x != 0 || y != 0 || z != 0;
        if has_movement {
            reset = true;
            let dir = Vec3::new(x as f32, y as f32, z as f32).normalize_or_zero();
            let speed = 5.0;
            uniforms.origin += time.delta_seconds() * speed * dir;
        }
    };

    let shader: Handle<Shader> = asset_server.get_handle("pathtracing.wgsl");
    match asset_server.get_load_state(shader) {
        bevy::asset::LoadState::Loading => {
            if *is_loading == false {
                info!("Reloading shader!");
                *is_loading = true;
            }
        }
        bevy::asset::LoadState::Loaded => {
            if *is_loading {
                *is_loading = false;
                reset = true;
            }
        }
        _ => (),
    }

    if reset {
        uniforms.frame = 0.0;
        *timer = time.seconds_since_startup();
        uniforms.time = 0.0;
    } else {
        uniforms.time = (time.seconds_since_startup() - *timer) as f32;
    }
}

const SIZE: (u32, u32) = (1280, 720);

#[derive(Component)]
struct ResizeSprite;

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut images: ResMut<Assets<Image>>,
) {
    asset_server.watch_for_changes().unwrap();

    let postprocessed_image = {
        let mut image = Image::new_fill(
            Extent3d {
                width: SIZE.0,
                height: SIZE.1,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            TextureFormat::Rgba32Float,
        );

        image.texture_descriptor.usage = TextureUsages::COPY_DST
            | TextureUsages::STORAGE_BINDING
            | TextureUsages::TEXTURE_BINDING;
        images.add(image)
    };

    let mut image = Image::new_fill(
        Extent3d {
            width: SIZE.0,
            height: SIZE.1,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        TextureFormat::Rgba32Float,
    );

    image.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
    let image = images.add(image);

    commands.insert_resource(PathTracingImage(image.clone(), postprocessed_image.clone()));
    commands.insert_resource(PathTracingUniform {
        size: Vec2::new(SIZE.0 as f32, SIZE.1 as f32),
        frame: -1.0,
        time: 0.0,
        origin: Vec3::ZERO,
    });

    commands.spawn_bundle(SpriteBundle {
        sprite: Sprite {
            custom_size: Some(Vec2::new(SIZE.0 as f32, SIZE.1 as f32)),
            ..default()
        },
        texture: postprocessed_image,
        ..default()
    }).insert(ResizeSprite);

    commands.spawn_bundle(Camera2dBundle {
        camera_2d: Camera2d {
            clear_color: ClearColorConfig::None,
            ..default()
        },
        ..default()
    });
}

fn update_image_to_window_size(
    windows: Res<Windows>,
    mut images: ResMut<Assets<Image>>,
    pathtracing_images: ResMut<PathTracingImage>,
    mut resize_events: EventReader<WindowResized>,
    mut uniforms: ResMut<PathTracingUniform>,
    mut sprites: Query<&mut Sprite, With<ResizeSprite>>,
) {
    for resize_event in resize_events.iter() {
        if !resize_event.id.is_primary() {
            continue;
        }

        let size = {
            let window = windows.get(resize_event.id).expect("Primary window");
            Extent3d {
                width: window.physical_width(),
                height: window.physical_height(),
                ..Default::default()
            }
        };

        uniforms.size = Vec2::new(size.width as f32, size.height as f32);
        sprites.single_mut().custom_size = Some(uniforms.size);

        {
            let image = images.get_mut(&pathtracing_images.0).expect("image");
            image.resize(size);
        }

        {
            let image = images.get_mut(&pathtracing_images.1).expect("image");
            image.resize(size);
        }
    }
}
