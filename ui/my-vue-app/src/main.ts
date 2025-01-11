import { createApp } from "vue";
import Main from "./Main.vue";
import App from "./Home.vue";
import Home from "./Dashboard.vue";
import Review from "./Review.vue";
import "./assets/index.css";

import { createWebHistory, createRouter } from "vue-router";

const routes = [
  { path: "/", component: App },
  { path: "/home", component: Home },
  { path: "/review", component: Review },
];

const router = createRouter({
  history: createWebHistory(),
  routes,
});

const main = createApp(Main);
main.use(router);
main.mount("#app");
