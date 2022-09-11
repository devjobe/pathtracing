let PI: f32 = 3.1415926535897932384626433832795;
let HALF_FOV = 0.78539816339744830961566084581988;
let F32_MAX = 3.40282347e+38;
let TOO_FAR = 10000.0;
let BOUNCES: u32 = 8u;
let SAMPLES: i32 = 32;
let EXPOSURE: f32 = 0.5;

@group(0) @binding(0)
var accumlated_texture: texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(1)
var post_texture: texture_storage_2d<rgba32float, read_write>;

struct PathTracingUniform
{
    size: vec2<f32>,
    frame: f32,
    time: f32,

    origin: vec3<f32>,
};

@group(1) @binding(0)
var<uniform> param : PathTracingUniform;


fn rand_pcg(state: ptr<function, u32>) -> u32
{
    let rnd: u32 = *state;
    *state = rnd * 747796405u + 2891336453u;
    let word: u32 = ((rnd >> ((rnd >> 28u) + 4u)) ^ rnd) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand_unorm(state: ptr<function, u32>) -> f32
{
    // 4294967040u?
    return f32(rand_pcg(state)) / 4294967296.0;
}

fn rand_snorm(state: ptr<function, u32>) -> f32
{
    return rand_unorm(state) * 2.0 - 1.0;
}

fn rand_unit(state: ptr<function, u32>) -> vec3<f32>
{
    let z = rand_snorm(state);
    let a = rand_unorm(state) * (2.0 * PI);
    let r = sqrt(1.0 - z * z);
    let x = r * cos(a);
    let y = r * sin(a);
    return vec3(x, y, z);
}

struct Ray
{
    origin: vec3<f32>,
    direction: vec3<f32>,
};

fn ray_point(ray: Ray, t: f32) -> vec3<f32>
{
    return ray.origin + ray.direction * t;
}

struct SceneHit
{
    normal: vec3<f32>,
    distance: f32,
    albedo: vec3<f32>,
    emissive: vec3<f32>,
    specular: vec3<f32>,
    specular_chance: f32,
    specular_roughness: f32,
    ior: f32,
    refraction_chance: f32,
    refraction_roughness: f32,
    refraction: vec3<f32>,
    inside_hit: bool,
};


fn raycast_plane(ray: Ray, current: ptr<function, SceneHit>, plane: vec4<f32>) -> bool
{
    let d = -(dot(ray.origin, plane.xyz)+plane.w)/dot(ray.direction, plane.xyz);
    if(d < 0.01 || d > TOO_FAR) {
        return false;
    }
    (*current).distance = d;
    (*current).normal = plane.xyz;
    (*current).inside_hit = false;
    return true;
}

fn raycast_tri(ray: Ray, current: ptr<function, SceneHit>, tri: array<vec3<f32>, 3>) -> bool
{
    let edge1 = tri[1] - tri[0];
    let edge2 = tri[2] - tri[0];

    var normal = cross(edge2, edge1);
    var nd = dot(normal, ray.direction);
    if(nd > 0.0) {
        normal = -normal;
        nd = -nd;
    }

    if(nd > -0.00001)
    {
        return false;
    }

    let dist = dot(normal, tri[0] - ray.origin) / nd;
    if(dist < 0.01 || dist > (*current).distance) {
        return false;
    }

    let perp = cross(ray.direction, edge2);
    let inv_det = 1.0 / dot(edge1, perp);

    let tvec = ray.origin - tri[0];
    let u = dot(tvec, perp) * inv_det;
    if(u < 0.0 || u > 1.0) {
        return false;
    }

    let qvec = cross(tvec, edge1);
    let v = dot(ray.direction, qvec) * inv_det;
    if(v < 0.0 || u + v > 1.0) {
        return false;
    }
   
    (*current).distance = dist;
    (*current).normal = normalize(normal);
    (*current).inside_hit = false;
    return true;
}

fn in_tri(edge1: vec3<f32>, edge2: vec3<f32>, tvec: vec3<f32>, dir: vec3<f32>) -> bool
{    
    let perp = cross(dir, edge2);
    let inv_det = 1.0 / dot(edge1, perp);

    let u = dot(tvec, perp) * inv_det;
    if(u < 0.0 || u > 1.0) {
        return false;
    }

    let qvec = cross(tvec, edge1);
    let v = dot(dir, qvec) * inv_det;
    if(v < 0.0 || u + v > 1.0) {
        return false;
    }

    return true;
}

fn raycast_quad(ray: Ray, current: ptr<function, SceneHit>, quad: array<vec3<f32>, 4>) -> bool
{

    let edge1 = quad[1] - quad[0];
    let edge2 = quad[2] - quad[0];

    var normal = cross(edge2, edge1);
    var nd = dot(normal, ray.direction);
    if(nd > 0.0) {
        normal = -normal;
        nd = -nd;
    }

    if(nd > -0.00001)
    {
        return false;
    }

    let dist = dot(normal, quad[0] - ray.origin) / nd;
    if(dist < 0.01 || dist > (*current).distance) {
        return false;
    }

    let tvec = ray.origin - quad[0];

    if(!in_tri(edge1, edge2, tvec, ray.direction)) {
        let edge3 = quad[3] - quad[0];
        if(!in_tri(edge3, edge2, tvec, ray.direction)) {
            return false;
        }
    }

    (*current).distance = dist;
    (*current).normal = normalize(normal);
    (*current).inside_hit = false;
    return true;
}

fn raycast_sphere(ray: Ray, current: ptr<function, SceneHit>, sphere: vec4<f32>) -> bool
{
    let m = ray.origin - sphere.xyz;
    let b = dot(m, ray.direction);
    let c = dot(m, m) - sphere.w * sphere.w;

    let is_outside = c > 0.0;
    let is_pointing_away = b > 0.0;

    if(is_outside && is_pointing_away)
    {
        return false;
    }

    let discr = b * b - c;
    if(discr < 0.0)
    {
        return false;
    }

    let d = sqrt(discr);

    var distance: f32 = -b - d;
    let is_inside = distance < 0.0;
    if(is_inside)
    {
        distance = -b + d;
    }
    if((*current).distance <= distance)
    {
        return false;
    }


    var normal = normalize((ray.origin+ray.direction * distance) - sphere.xyz);

    if(is_inside)
    {
        normal = -normal;
    }

    (*current).normal = normal;
    (*current).distance = distance;
    (*current).inside_hit = is_inside;
    return true;
}

fn scene_trace(ray: Ray, scene_hit: ptr<function, SceneHit>) -> bool
{   
    (*scene_hit).albedo = vec3(0.7);
    (*scene_hit).emissive = vec3(0.0);
    (*scene_hit).specular = vec3(0.0);
    (*scene_hit).specular_chance = 0.0;
    (*scene_hit).specular_roughness = 0.0;
    (*scene_hit).ior = 1.0;
    (*scene_hit).refraction_chance = 0.0;
    (*scene_hit).refraction_roughness = 0.0;
    (*scene_hit).refraction = vec3(0.0);

    // back wall
    {
        let A = vec3<f32>(-12.6, -12.6, 35.0);
        let B = vec3<f32>( 12.6, -12.6, 35.0);
        let C = vec3<f32>( 12.6, 12.6, 35.0);
        let D = vec3<f32>(-12.6, 12.6, 35.0);
        if (raycast_quad(ray, scene_hit, array<vec3<f32>, 4>(A, B, C, D)))
        {
        }
    }

    // floor
    {
        let A = vec3<f32>(-12.6, -12.45, 35.0);
        let B = vec3<f32>( 12.6, -12.45, 35.0);
        let C = vec3<f32>( 12.6, -12.45, 25.0);
        let D = vec3<f32>(-12.6, -12.45, 25.0);
        if (raycast_quad(ray, scene_hit, array<vec3<f32>, 4>(A, B, C, D)))
        {
        }
    }
    
    // ceiling
    {
        let A = vec3<f32>(-12.6, 12.5, 35.0);
        let B = vec3<f32>( 12.6, 12.5, 35.0);
        let C = vec3<f32>( 12.6, 12.5, 25.0);
        let D = vec3<f32>(-12.6, 12.5, 25.0);
        if (raycast_quad(ray, scene_hit, array<vec3<f32>, 4>(A, B, C, D)))
        {
        }
    }

    // left
    {
        let A = vec3<f32>(-12.5, -12.6, 35.0);
        let B = vec3<f32>(-12.5, -12.6, 25.0);
        let C = vec3<f32>(-12.5, 12.6, 25.0);
        let D = vec3<f32>(-12.5, 12.6, 35.0);
        if (raycast_quad(ray, scene_hit, array<vec3<f32>, 4>(A, B, C, D)))
        {
            (*scene_hit).albedo = vec3(0.1, 0.7, 0.1);
        }
    }

    // right
    {
        let A = vec3<f32>(12.5, -12.6, 35.0);
        let B = vec3<f32>(12.5, -12.6, 25.0);
        let C = vec3<f32>(12.5, 12.6, 25.0);
        let D = vec3<f32>(12.5, 12.6, 35.0);
        if (raycast_quad(ray, scene_hit, array<vec3<f32>, 4>(A, B, C, D)))
        {
            (*scene_hit).albedo = vec3(0.7, 0.1, 0.1);
        }
    }

    // light
    {
        let A = vec3<f32>(-5.0, 12.4,  32.5);
        let B = vec3<f32>( 5.0, 12.4,  32.5);
        let C = vec3<f32>( 5.0, 12.4,  27.5);
        let D = vec3<f32>(-5.0, 12.4,  27.5);
        if (raycast_quad(ray, scene_hit, array<vec3<f32>, 4>(A, B, C, D)))
        {
            (*scene_hit).albedo = vec3(0.0);
            (*scene_hit).emissive = vec3(1.0, 0.9, 0.7) * 20.0;
        }
    }


    if(raycast_sphere(ray, scene_hit, vec4<f32>(9.0, -9.5, 30.0, 3.0)))
    {
        (*scene_hit).albedo = vec3(0.0, 0.0, 1.0);
        (*scene_hit).emissive = vec3(0.0);
        (*scene_hit).specular = vec3(1.0, 0.0, 0.0);
        (*scene_hit).specular_chance = 0.5;
        (*scene_hit).specular_roughness = 0.2;
    }

    if(raycast_sphere(ray, scene_hit, vec4<f32>(-9.0, -9.5, 30.0, 3.0)))
    {
        (*scene_hit).albedo = vec3(0.9, 0.9, 0.5);
        (*scene_hit).emissive = vec3(0.0);
        (*scene_hit).specular = vec3(0.9);
        (*scene_hit).specular_chance = 1.0;
        (*scene_hit).specular_roughness = 0.2;
    }

    if(raycast_sphere(ray, scene_hit, vec4<f32>(0.0, -9.5, 30.0, 3.0)))
    {
        (*scene_hit).albedo = vec3(0.9, 0.5, 0.9);
        (*scene_hit).emissive = vec3(0.0);
        (*scene_hit).specular = vec3(0.9);
        (*scene_hit).specular_chance = 0.3;
        (*scene_hit).specular_roughness = 0.2;

        (*scene_hit).refraction_chance = 1.0;
        (*scene_hit).refraction_roughness = 0.2;
        // (*scene_hit).refraction = vec3(1.0);
    }
    // roughness

    let count = 8;
    for(var i: i32 = 0;i < count;i++)
    {
        if(raycast_sphere(ray, scene_hit, vec4<f32>(-10.5 + f32(i*3), 0.0, 33.0, 1.25)))
        {
            let t = f32(i) / f32(count - 1);
            (*scene_hit).albedo = vec3(0.9, 0.25, 0.25);
            (*scene_hit).emissive = vec3(0.0);
            (*scene_hit).specular = vec3(0.8);
            (*scene_hit).specular_chance = 0.02;
            (*scene_hit).specular_roughness = 0.0;
            (*scene_hit).ior = 1.1;
            (*scene_hit).refraction_chance = 1.0;
            (*scene_hit).refraction_roughness = 0.1;
            (*scene_hit).refraction = vec3(1.0, 2.0, 3.0) * t * 2.0;
       }
    }


    let d = (*scene_hit).distance;
    return d > 0.01 && d < TOO_FAR;
}

fn fresnel_amount(n1: f32, n2: f32, normal: vec3<f32>, incident: vec3<f32>, f0: f32, f90: f32) -> f32
{
    var cosX = -dot(normal, incident);
    if(n1 > n2) {
        let n = n1 / n2;
        let sinT2 = n*n*(1.0-cosX*cosX);
        if(sinT2 > 1.0) {
            return f90;
        }
        cosX = sqrt(1.0 - sinT2);
    }
    let k = (n1 - n2) / (n1 + n2);
    let r0 = k * k;
    let x = clamp(1.0-cosX, 0.0, 1.0);
    let t = r0+(1.0-r0)*x*x*x*x*x;
    return mix(f0, f90, t);
}

fn get_refract(x: vec3<f32>, y: vec3<f32>, z: f32) -> vec3<f32>
{
    let dp = dot(x,y);
    let k = 1.0 - z * z * (1.0 - dp * dp);
    if(k < 0.0) {
        return vec3(0.0);
    }
    return z * x - (z * dp + sqrt(k)) * y;
}

fn get_refraction(direction: vec3<f32>, scene_hit: ptr<function, SceneHit>, rnd: ptr<function, u32>) -> vec3<f32>
{
    let ior: f32 = select(1.0 / (*scene_hit).ior, (*scene_hit).ior, (*scene_hit).inside_hit);
    let refraction : vec3<f32> = get_refract(direction, (*scene_hit).normal, ior);
    let d = normalize(-(*scene_hit).normal + rand_unit(rnd));
    let refraction_direction = normalize(mix(refraction, d, (*scene_hit).refraction_roughness * (*scene_hit).refraction_roughness));        
    return refraction_direction;
}

fn sample_scene(input_ray: Ray, rnd: ptr<function, u32>) -> vec3<f32>
{    
    var scene_hit : SceneHit;
    var radiance = vec3(0.0);
    var throughput = vec3<f32>(1.0);
    var ray: Ray = input_ray;
    for(var i: u32 = 0u;i < BOUNCES;i++)
    {
        scene_hit.distance = F32_MAX;
        if(!scene_trace(ray, &scene_hit))
        {
            radiance += vec3(0.0, 0.07, 0.25) * throughput;
            break;
        }

        if (scene_hit.inside_hit) {
            throughput *= exp(-scene_hit.refraction * scene_hit.distance);
        }

        radiance += scene_hit.emissive * throughput;

        var specular_chance = scene_hit.specular_chance;
        var refraction_chance = scene_hit.refraction_chance;
        if(specular_chance > 0.0)
        {
            let n1 = select(scene_hit.ior, 1.0, !scene_hit.inside_hit);
            let n2 = select(scene_hit.ior, 1.0, scene_hit.inside_hit);
            specular_chance = fresnel_amount(n1, n2, ray.direction, scene_hit.normal, specular_chance, 1.0);
            refraction_chance *= (1.0 - specular_chance) / (1.0 - scene_hit.specular_chance);
        }

        let roll = rand_unorm(rnd);

        var is_specular: f32 = 0.0;
        var is_refractive: f32 = 0.0;
        var probablility: f32;

        if(specular_chance > 0.0 && roll < specular_chance)
        {
            is_specular = 1.0;
            probablility = specular_chance;
        }
        else if(refraction_chance > 0.0 && roll < refraction_chance + specular_chance)
        {
            is_refractive = 1.0;
            probablility = refraction_chance;
        }
        else
        {
            probablility = 1.0 - refraction_chance + specular_chance;
        }      


        if(is_refractive == 0.0) {
            ray.origin = ray.origin + ray.direction * scene_hit.distance + scene_hit.normal * 0.01;                        
            let luminance = mix(scene_hit.albedo, scene_hit.specular, is_specular);
            throughput *= luminance;
        }
        else
        {
            ray.origin = ray.origin + ray.direction * scene_hit.distance - scene_hit.normal * 0.01;                        
        }


        let diffuse_direction = normalize(scene_hit.normal + rand_unit(rnd));
        let reflect_direction = reflect(ray.direction, scene_hit.normal);
        let specular_direction = normalize(mix(reflect_direction, diffuse_direction, scene_hit.specular_roughness * scene_hit.specular_roughness));
        let refraction_direction = get_refraction(ray.direction, &scene_hit, rnd);
        ray.direction = mix(mix(diffuse_direction, specular_direction, is_specular), refraction_direction, is_refractive);
        
        throughput *= 1.0 / max(probablility, 0.001);

        let early_break_chance = max(throughput.r, max(throughput.g, throughput.b));
        if(rand_unorm(rnd) > early_break_chance) {
            break;
        }
        throughput *= 1.0 / early_break_chance;
    }

    return radiance;
}

fn get_ray_target(uv: vec2<f32>) -> vec3<f32>
{
    let aspect = param.size.y / param.size.x;
    let camera_distance = 1.0 / tan(HALF_FOV);
    let ray_target = vec3<f32>(uv.x * 2.0 - 1.0, (uv.y * 2.0 - 1.0) * aspect, camera_distance);    
    return ray_target;
}

fn compute_pixel(invocation_id: vec3<u32>) -> vec3<f32>
{
    let color: vec4<f32> = vec4<f32>(0.0);
    var rnd: u32 = u32(invocation_id.x * 1973u + invocation_id.y * 9277u) + u32(param.frame) * 747796405u;

    let jitter = vec2<f32>(rand_unorm(&rnd), rand_unorm(&rnd)) - 0.5;
	var coord = vec2<f32>(f32(invocation_id.x), f32(invocation_id.y)) + jitter;
    var uv: vec2<f32> = coord / param.size;


    var ray = Ray(param.origin, normalize(get_ray_target(uv)));
    var color: vec3<f32> = vec3(0.0);
    for(var i: i32 = 0;i < SAMPLES;i++)
    {
        color += sample_scene(ray, &rnd) / f32(SAMPLES);
    }    

    return color;
}

fn srgb_from_unorm(rgb: vec3<f32>) -> vec3<f32>
{
    let a = pow(rgb, vec3<f32>(1.0 / 2.4)) * 1.055 - 0.055;
    let b = rgb * 12.92;
    let t = step(rgb, vec3<f32>(0.0031308));
    return mix(a, b, t);
}

fn unorm_aces_film(x: vec3<f32>) -> vec3<f32>
{
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x*(a*x + b)) / (x*(c*x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn process_invocation(invocation_id: vec3<u32>, accumlate: bool)
{
    let inverted_y = vec2<i32>(i32(invocation_id.x), i32(param.size.y) - i32(invocation_id.y));
    var color = compute_pixel(invocation_id);

    if (accumlate) {
        var current = textureLoad(accumlated_texture, inverted_y);
        color = mix(current.xyz, color, 1.0 / param.frame);
    }

    textureStore(accumlated_texture, inverted_y, vec4<f32>(color, 1.0));

    color = unorm_aces_film(color * EXPOSURE);
    color = srgb_from_unorm(color);
    textureStore(post_texture, inverted_y, vec4<f32>(color, 1.0));

}

@compute @workgroup_size(8, 8, 1)
fn init(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    process_invocation(invocation_id, false);
}

@compute @workgroup_size(8, 8, 1)
fn update(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    process_invocation(invocation_id, true);
}

