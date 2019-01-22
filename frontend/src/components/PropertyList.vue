<template>
  <v-data-table
    v-model="selected"
    :headers="headers"
    :items="items"
    select-all
    item-key="name"
    class="elevation-1 d-inline-block"
    hide-actions
  >
    <template slot="items" slot-scope="props">
      <tr :active="props.selected" @click="props.selected = !props.selected">
        <td>
          <v-checkbox
            :input-value="props.selected"
            primary
            hide-details
          ></v-checkbox>
        </td>
        <td v-for="key in Object.keys(props.item)"
            :key="key"
        >
            {{ props.item[key] }}
        </td>
      </tr>
    </template>
  </v-data-table>
</template>

<script>
export default {
    props: {
        headers: Array,
        items: Array,
        selected: Array,
    },
    data: () => ({
        headers2: [
            {
                text: 'Dessert (100g serving)',
                align: 'left',
                sortable: false,
                value: 'name',
            },
        ],
        desserts: [
            {
                name: 'Frozen Yogurt',
            },
            {
                name: 'Ice cream sandwich',
            },
        ],
    }),
    methods: {
        toggleAll() {
            if (this.selected.length) this.selected = [];
            else this.selected = this.desserts.slice();
        },
    },
};
</script>

<style>
</style>
