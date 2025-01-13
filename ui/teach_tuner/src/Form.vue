<script setup lang="ts">
import { Button } from "@/components/ui/button";
import {
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/app/ui/form";
import { Input } from "@/app/ui/input";
import Toaster from "@/app/ui/toast/Toaster.vue";
import { useToast } from "@/app/ui/toast/use-toast";
import { Switch } from "@/app/ui/switch";
import { Textarea } from "@/app/ui/textarea";
import { vAutoAnimate } from "@formkit/auto-animate/vue";

import { toTypedSchema } from "@vee-validate/zod";
import { useForm } from "vee-validate";
import { h } from "vue";
import * as z from "zod";

const { toast } = useToast();

const formSchema = toTypedSchema(
  z.object({
    activity_name: z.string().min(2).max(50),
    description: z
      .string()
      .min(10, {
        message: "description must be at least 10 characters.",
      })
      .max(160, {
        message: "description must not be longer than 30 characters.",
      }),
    allow_visibility: z.boolean().default(true),
  }),
);

const { isFieldDirty, handleSubmit } = useForm({
  validationSchema: formSchema,
});

const onSubmit = handleSubmit(async (values) => {
  try {
    console.log(JSON.stringify(values));

    // Effettua la richiesta POST all'endpoint FastAPI con fetch
    const response = await fetch("http://localhost:8080/save-data/", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(values),
    });

    // Controlla se la risposta è positiva
    if (!response.ok) {
      throw new Error("Network response was not ok");
    }

    // Estrai i dati dalla risposta
    const data = await response.json();
    const { message, id } = data;

    // Mostra una notifica con il risultato
    toast({
      title: "Data submitted successfully!",
      description: h(
        "div",
        { class: "mt-2 w-[340px] rounded-md bg-slate-950 p-4" },
        [
          h("p", { class: "text-white" }, `Message: ${message}`),
          h("p", { class: "text-white mt-2" }, `Generated ID: ${id}`),
        ],
      ),
    });
  } catch (error) {
    // Gestione errori: mostra un messaggio di errore
    toast({
      title: "Submission failed",
      description: h(
        "pre",
        { class: "mt-2 w-[340px] rounded-md bg-red-950 p-4" },
        h("code", { class: "text-white" }, error.message),
      ),
    });
  }
});
</script>

<template>
  <Toaster />
  <div class="m-4 h-screen">
    <form
      class="w-full space-y-6 max-w-xl bg-white p-8 rounded-lg border bg-card text-card-foreground shadow-sm"
      @submit="onSubmit"
    >
      <div>
        <h1 class="text-3xl">Forms Builder</h1>
      </div>

      <FormField
        v-slot="{ componentField }"
        name="activity_name"
        :validate-on-blur="!isFieldDirty"
      >
        <FormItem v-auto-animate>
          <FormLabel>Activity Name</FormLabel>
          <FormControl>
            <Input type="text" placeholder="Example" v-bind="componentField" />
          </FormControl>
          <FormDescription>
            This is the name of the activity to be reviewed.
          </FormDescription>
          <FormMessage />
        </FormItem>
      </FormField>

      <FormField
        v-slot="{ componentField }"
        name="description"
        :validate-on-blur="!isFieldDirty"
      >
        <FormItem v-auto-animate>
          <FormLabel>Description</FormLabel>
          <FormControl>
            <Textarea
              placeholder="Describe the aspects of the activity you want reviewed"
              class="resize-none"
              v-bind="componentField"
            />
          </FormControl>
          <FormDescription>
            You can insert <span>@mention</span> to other activity.
          </FormDescription>
          <FormMessage />
        </FormItem>
      </FormField>

      <FormField v-slot="{ value, handleChange }" name="allow_visibility">
        <FormItem
          class="flex flex-row items-center justify-between rounded-lg border p-4"
        >
          <div class="space-y-0.5 mr-3">
            <FormLabel class="text-base"> Form Visibility </FormLabel>
            <FormDescription>
              Allow all people who have the code to view the form
            </FormDescription>
          </div>
          <FormControl>
            <Switch
              :default-checked="true"
              :checked="value"
              @update:checked="handleChange"
            />
          </FormControl>
        </FormItem>
      </FormField>

      <Button class="w-full" type="submit">Submit</Button>
    </form>
  </div>
</template>
