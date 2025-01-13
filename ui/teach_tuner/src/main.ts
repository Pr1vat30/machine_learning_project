import { createApp } from "vue";
import Main from "./Main.vue";
import Home from "./Home.vue";
import Dashboard from "./Dashboard.vue";
import Review from "./Review.vue";
import "./assets/index.css";

import { createWebHistory, createRouter } from "vue-router";

const routes = [
  { path: "/", component: Home },
  { path: "/review", component: Review },
  { path: "/dashboard", component: Dashboard },
];

const router = createRouter({
  history: createWebHistory(),
  routes,
});

const main = createApp(Main);
main.use(router);
main.mount("#app");
