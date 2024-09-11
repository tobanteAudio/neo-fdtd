// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2021 Brian Hamilton

///////////////////////////////////////////////////////////////////////////////
// This file is a part of PFFDTD.
//
// PFFTD is released under the MIT License.
// For details see the LICENSE file.
//
// Copyright 2021 Brian Hamilton.
//
// File name: common.js
//
// Description: some functions to help with setting ThreeJS scene
//
///////////////////////////////////////////////////////////////////////////////

function onWindowResize(obj) {
  let scene_container = obj.scene_container;
  let camera = obj.camera;
  let renderer = obj.renderer;

  if (!scene_container) {
    return;
  }
  console.log("You resized the browser window!");
  // set the aspect ratio to match the new browser window aspect ratio
  camera.aspect = scene_container.clientWidth / scene_container.clientHeight;

  // update the camera's frustum
  camera.updateProjectionMatrix();

  // update the size of the renderer AND the canvas
  renderer.setSize(scene_container.clientWidth, scene_container.clientHeight);
}

async function createScene(obj) {
  // Get a reference to the container element that will hold our scene
  obj.scene_container = document.querySelector("#scene-container");
  let scene_container = obj.scene_container;

  if (!obj.scene) {
    // create a Scene
    obj.scene = new THREE.Scene();
    let scene = obj.scene;

    // Set the background color
    scene.background = new THREE.Color("skyblue"); //0x87CEEB
  }
}
async function createRenderer(obj, update) {
  let scene_container = obj.scene_container;

  let renderer = obj.renderer;
  //console.log(renderer);
  if (!renderer) {
    // create the renderer (canvas)
    obj.renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer = obj.renderer;

    renderer.setSize(scene_container.clientWidth, scene_container.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);

    renderer.gammaFactor = 2.2;
    //renderer.gammaOutput = true;

    // add the automatically created <canvas> element to the page
    scene_container.appendChild(renderer.domElement);
  }

  // start the animation loop
  renderer.setAnimationLoop(() => {
    update();
    render(obj);
  });
}

async function createStats(obj, update) {
  let scene_container = obj.scene_container;
  let renderer = obj.renderer;
  let stats = obj.stats;
  if (!renderer) {
    return;
  }

  obj.stats = new Stats();
  stats = obj.stats;
  stats.domElement.style.cssText = "position:absolute;top:0px;right:0px;";
  //scene_container.appendChild( stats.domElement );
  scene_container.appendChild(stats.domElement);
  // resstart the animation loop
  renderer.setAnimationLoop(() => {
    update();
    stats.begin();
    render(obj);
    stats.end();
  });
}
function render(obj) {
  let renderer = obj.renderer;
  let scene = obj.scene;
  let camera = obj.camera;
  renderer.render(scene, camera);
}

function sph2cart(az, el) {
  x = Math.cos((Math.PI / 180.0) * el) * Math.cos((Math.PI / 180.0) * az);
  y = Math.cos((Math.PI / 180.0) * el) * Math.sin((Math.PI / 180.0) * az);
  z = Math.sin((Math.PI / 180.0) * el);
  return { x: x, y: y, z: z };
}
